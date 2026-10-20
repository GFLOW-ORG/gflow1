"""
Particle tracing checks for a well in a uniform flow field.

These follow the code verification for a well in a uniform flow in
Haitjema, "WhAEM Verifications", section 1.1, which uses the same model as
test/data/well_uflow.dat. Units are meters and days.

    k = 10 m/d, H = 10 m, n = 0.2, Q = 50 m^3/d, rw = 1 m,
    Q0 = 1 m^2/d in the x-direction, head 100 m at the well (0, 0).

Checked against the analytic solution:

* the stagnation point downstream of the well, xs = Q / (2 pi Q0) = 7.957 m;
* the capture zone width opposite the well, Q / (2 Q0) = 25 m (half-width
  12.5 m), see also Haitjema (1995), Analytic Element Modeling of Groundwater
  Flow, eq. (3.116);
* the travel time along the negative x-axis, WhAEM Verifications eq. (1.7).

A small tracing step (0.1 m) is used: with the default step, a particle
starting 0.1 m inside the capture zone boundary is not captured.
"""

import math
import os
import pathlib
import re
import subprocess

import pytest

# Parameters from test/data/well_uflow_trace.dat [m, d]
N = 0.2  # porosity
H = 10.0  # aquifer thickness [m]
Q0 = 1.0  # uniform flow rate in the x-direction [m^2/d]
Q = 50.0  # well discharge [m^3/d]
XS = Q / (2.0 * math.pi * Q0)  # distance well to stagnation point [m]

FLOAT = r"([-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?)"
END_LINE = re.compile(
    r"^\+end\s+xyz\s+" + r"\s+".join([FLOAT] * 3) + r"\s+end\s+time\s+" + FLOAT
)


def travel_time(d0, d1):
    """
    Travel time [d] along the negative x-axis from distance d0 to distance d1
    upstream of the well, WhAEM Verifications eq. (1.7).
    """

    def f(d):
        return d - XS * math.log(d + XS)

    return N * H / Q0 * (f(d0) - f(d1))


def distance_after(t, d0):
    """Distance upstream of the well after travelling t days from d0 (bisection)."""
    lo, hi = 1.0, d0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if travel_time(d0, mid) > t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.fixture(scope="module")
def ends(tmp_path_factory):
    """Run gflow2 once and return the end point (x, y, z, t) of each streamline."""
    tmpdir = tmp_path_factory.mktemp("trace")
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    os.environ["PATH"] += os.pathsep + str(project_root / "bin")

    # gflow1 upper-cases file names and does not accept input file names
    # longer than 16 characters, so use a short name.
    source = project_root / "test/data/well_uflow_trace.dat"
    (tmpdir / "UFLOW_TR.DAT").write_bytes(source.read_bytes())

    subprocess.run(
        ["gflow2", "UFLOW_TR.DAT"],
        cwd=tmpdir,
        timeout=10,
        check=True,
        capture_output=True,
        text=True,
    )

    message = (tmpdir / "TRACE-MSG.LOG").read_text()
    result = []
    for line in message.splitlines():
        match = END_LINE.match(line)
        if match:
            result.append(tuple(float(v.replace("D", "E")) for v in match.groups()))
    # One particle in the first trace, six in the second.
    assert len(result) == 7
    return result


def captured(end):
    """True if the particle ended at the well (rw = 1 m)."""
    x, y, _, _ = end
    return math.hypot(x, y) < 1.5


def escaped(end):
    """True if the particle left the window downstream (x = 25 m)."""
    x, _, _, _ = end
    return x > 24.0


def test_stagnation_point(ends):
    assert XS == pytest.approx(7.957, abs=1e-3)
    # Starts just upstream of the stagnation point: flows back into the well.
    assert captured(ends[1])
    # Starts just downstream of the stagnation point: flows away from the well.
    assert escaped(ends[2])


def test_capture_zone_width_opposite_well(ends):
    half_width = Q / (4.0 * Q0)  # 12.5 m
    assert half_width == pytest.approx(12.5)
    # Starting at (0, +-12.4), just inside the capture zone.
    assert captured(ends[3])
    assert captured(ends[4])
    # Starting at (0, +-12.6), just outside the capture zone.
    assert escaped(ends[5])
    assert escaped(ends[6])


def test_travel_time_along_axis(ends):
    # The WhAEM Verifications result: 347.95 days from x = -200 m to the well.
    assert travel_time(200.0, 1.0) == pytest.approx(347.95, abs=0.01)
    # The particle from (-200, 0) is traced for (up to) 300 days. Compare its
    # end point with the analytic position at the reported end time.
    x, y, _, t = ends[0]
    assert t == pytest.approx(300.0, abs=0.1)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert x == pytest.approx(-distance_after(t, 200.0), abs=0.05)
