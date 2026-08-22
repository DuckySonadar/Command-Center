"""Check the fish designer's JS copy of the ring cutter against joint_tool.py.

`tools/fish-editor-nurbs.html` carries a full port of the solid so the browser
can build the same printable plate the generator does. Two files describing one
piece of geometry drift, and quietly: a placement that is off by a millimetre
still meshes, still looks like a joint, and comes off the printer fused. This
holds one to the other.

    python3 check_tool_port.py

Three things are compared. The two fields are sampled on random points
rather than a lattice, so nothing lands on a symmetry by luck:

  raw       the solid in its own frame -- the node list and the blends
  placed    the solid as a joint uses it: scaled to a section, moved to `xa`,
            and displaced by tool_offset / tool_lift / tool_scale
  joints    the placement pipeline itself, on the default fish: every joint's
            cut, tool position, lift, three scales, clearance and footprint,
            with per-joint `tool_place` entries in play. The field maths being
            identical does not help if the two sides disagree about where to
            put the thing.

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
TOL_RAW, TOL_PLACED, TOL_JOINT = 1e-12, 1e-4, 1e-5


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
const face = Object.assign({}, NurbsCore.DEFAULTS, NurbsCore.FIXED, P.face);
const C = NurbsCore.prepare(P.shape, face, false);
console.log(JSON.stringify({ field: out, joints: C.joints,
                             errors: C.errors }));
"""

# what a joint has to agree about, and what it is called on each side
JOINT_KEYS = [("xa", "xa"), ("xt", "xt"), ("lift", "lift"),
              ("s_wide", "sWide"), ("s_long", "sLong"), ("s_tall", "sTall"),
              ("gap", "gap"), ("ahead", "ahead"), ("behind", "behind")]


def main():
    if not shutil.which("node"):
        print("no node on PATH -- skipping")
        return 0
    src, where = core_source()
    if src is None:
        print("no fish-editor-nurbs.html in the tree or on main -- skipping")
        return 0

    rng = np.random.default_rng(11)
    # the solid's own frame, with a margin: x -21.2 .. 5.6 about an anchor at
    # -10.2, y +/-12, z -7.6 .. 22 off a plate at 0
    raw = rng.uniform([-26, -16, -12], [11, 16, 27], size=(4000, 3))
    # one joint, placed the way a fish places it and then moved by hand
    jd = dict(xa=62.5, xt=62.5 + 7.5, lift=3.0,
              sWide=0.82, sLong=1.15, sTall=1.15)
    gap = 0.9
    placed = rng.uniform([jd["xt"] - 15, -16, jd["lift"] - 11],
                         [jd["xt"] + 20, 16, jd["lift"] + 27], size=(4000, 3))

    # the placement pipeline, on the default fish. Both fins come out: a ring
    # joint reaches straight through the pelvic socket, and the cutter's box
    # reaches far enough behind a joint to shear the dorsal fin off as a loose
    # piece, so the generator refuses to build either -- see
    # FishBuilder._check_side_fins and _check_dorsal_fin. This check is about
    # where the solid gets *placed*, and finless is the configuration that
    # exercises every joint without the build stopping first.
    shape = {"curves": {"pelvic_fin": None, "dorsal_fin": None}}
    cfg = {"joint_style": "tool", "tool_offset": 3.5, "tool_lift": 1.25,
           "tool_scale": 0.9,
           "tool_place": [{"off": -6.0, "long": 1.3},
                          None,
                          {"lift": 2.5, "tall": 0.75, "off": 4.0}]}

    probe = os.path.join(HERE, "_tool_probe.js")
    args = os.path.join(HERE, "_tool_args.json")
    with open(probe, "w") as f:
        f.write(extract(src))
        f.write(PROBE)
    with open(args, "w") as f:
        json.dump({"raw": raw.tolist(), "placed": placed.tolist(),
                   "joint": jd, "gap": gap, "shape": shape, "face": cfg}, f)
    try:
        r = subprocess.run(["node", probe, args], capture_output=True,
                           text=True, check=True)
    finally:
        for path in (probe, args):
            os.remove(path)
    got = json.loads(r.stdout)
    theirs = np.array(got["field"])

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

    # ---- the placement pipeline ------------------------------------
    sys.path.insert(0, REPO)
    import flexifish_nurbs as fn                                # noqa: E402
    from flexifish import FishParams                            # noqa: E402
    mine = fn.NurbsFishBuilder(FishParams(**cfg), shape).joints
    if got["errors"]:
        print(f"  joints  the designer refused the test fish: {got['errors']}")
        return 1
    if len(mine) != len(got["joints"]):
        print(f"  joints  {len(mine)} here, {len(got['joints'])} there")
        return 1
    worst, where = 0.0, ""
    for i, (a, b) in enumerate(zip(mine, got["joints"])):
        for pk, jk in JOINT_KEYS:
            d = abs(float(a[pk]) - float(b[jk]))
            if d > worst:
                worst, where = d, f"joint {i} {pk}"
    # 1e-5, not machine epsilon: every scale here is `top_at(x)` divided by
    # a reference, and `top_at` probes a float32 grid on both sides. What this
    # is looking for is a placement that differs, which starts at a hundredth
    # of a millimetre, not at the last bit of a float.
    print(f"  joints  {len(mine)} joints, worst |err| {worst:.2e} "
          f"(tol {TOL_JOINT:.0e}){'   at ' + where if worst else ''}")
    bad += worst > TOL_JOINT
    print("\nMATCH" if not bad else "\nMISMATCH")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
