"""Check the fish designer's JS copy of the ring cutter against joint_tool.py.

`tools/fish-editor-nurbs.html` carries a full port of the solid so the browser
can build the same printable plate the generator does. Two files describing one
piece of geometry drift, and quietly: a placement that is off by a millimetre
still meshes, still looks like a joint, and comes off the printer fused. This
holds one to the other.

    python3 check_tool_port.py

Two things are compared, on random points rather than a lattice so nothing
lands on a symmetry by luck:

  raw       the solid in its own frame -- the node list and the blends
  placed    the solid as a joint uses it: scaled to a section, moved to `xa`,
            and displaced by tool_offset / tool_lift / tool_scale

The two are held to different tolerances on purpose. `raw` is compared in
double precision on both sides and has to agree to 1e-12 -- that is the same
arithmetic twice, and anything looser would hide a transcription slip.
`tool_sdf` casts to float32, because that is what the mesher wants, so its
agreement is a float32 one: about 1e-5 mm on a 50 mm field, and a tolerance
tight enough to catch a real difference is 1e-4.

Needs node on PATH. Skips (does not fail) if node or the designer is missing,
so it stays usable in a checkout that has neither.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, REPO)
import joint_tool as jt                                        # noqa: E402

SITE = os.environ.get("MAKERCAVE_REPO",
                      os.path.join(REPO, os.pardir,
                                   "mywebsiterepository-Iknowtotallyoriginal"))
EDITOR = os.path.join(SITE, "tools", "fish-editor-nurbs.html")
TOL_RAW, TOL_PLACED = 1e-12, 1e-4


def core_source():
    """The designer's core module, which is DOM-free and runnable under node.

    Taken from the working tree, or from the website repo's main branch if
    this checkout has no sibling for it."""
    if os.path.exists(EDITOR):
        return open(EDITOR).read(), EDITOR
    r = subprocess.run(["git", "-C", SITE, "show",
                        "origin/main:tools/fish-editor-nurbs.html"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout, f"{SITE}:origin/main"
    return None, None


def extract(src):
    a = src.find('<script id="coresrc">')
    if a < 0:
        raise SystemExit("no <script id=\"coresrc\"> in the designer")
    a = src.index("\n", a) + 1
    b = src.index("</script>", a)
    return src[a:b]


PROBE = """
const P = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const q of P.raw) out.push(NurbsCore.toolRaw(q[0], q[1], q[2]));
for (const q of P.placed)
  out.push(NurbsCore.toolSDF(q[0], q[1], q[2], P.joint, P.gap));
console.log(JSON.stringify(out));
"""


def main():
    if not shutil.which("node"):
        print("no node on PATH -- skipping")
        return 0
    src, where = core_source()
    if src is None:
        print("no fish-editor-nurbs.html in the tree or on main -- skipping")
        return 0

    rng = np.random.default_rng(11)
    # the tool's own frame: x across, y along, z up from its plate at 8
    raw = rng.uniform([-30, -6, 4], [30, 48, 54], size=(4000, 3))
    # one joint, placed the way a fish places it and then moved by hand
    jd = dict(xa=62.5, xt=62.5 + 7.5, lift=3.0,
              sWide=0.82, sLong=1.15, sTall=1.15)
    gap = 0.9
    placed = rng.uniform([jd["xt"] - 20, -30, -4], [jd["xt"] + 50, 30, 55],
                         size=(4000, 3))

    probe = os.path.join(HERE, "_tool_probe.js")
    args = os.path.join(HERE, "_tool_args.json")
    with open(probe, "w") as f:
        f.write(extract(src))
        f.write(PROBE)
    with open(args, "w") as f:
        json.dump({"raw": raw.tolist(), "placed": placed.tolist(),
                   "joint": jd, "gap": gap}, f)
    try:
        r = subprocess.run(["node", probe, args], capture_output=True,
                           text=True, check=True)
    finally:
        for path in (probe, args):
            os.remove(path)
    theirs = np.array(json.loads(r.stdout))

    print(f"vs {where}")
    bad = 0
    for name, tol, a, b in (
            ("raw", TOL_RAW,                       # both sides in float64
             jt.raw(raw[:, 0], raw[:, 1], raw[:, 2]), theirs[:len(raw)]),
            ("placed", TOL_PLACED,                 # tool_sdf is float32
             jt.tool_sdf(
                 *(placed[:, i].astype(jt.F32) for i in range(3)),
                 xa=jd["xt"], s_wide=jd["sWide"], s_long=jd["sLong"],
                 s_tall=jd["sTall"], lift=jd["lift"], gap=gap),
             theirs[len(raw):])):
        err = float(np.max(np.abs(a - b)))
        agree = float(np.mean((a < 0) == (b < 0)) * 100)
        print(f"  {name:7s} max |err| {err:.2e} (tol {tol:.0e})   "
              f"sign agreement {agree:.3f}%")
        bad += err > tol or agree < 100.0
    print("\nMATCH" if not bad else "\nMISMATCH")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
