"""Check `sdf_json.py` against the real evaluator in `sdf_editor.html`.

The editor is the authority on what a scene file means. Rather than trust a
reading of it, this pulls its own `PRIMS`, `smin`/`smax`, `invRot` and
`sceneSDF` straight out of the HTML, runs them under node on a few thousand
points, and compares against the Python. Any drift in either side shows up
here rather than in a mesh three steps later.

    python3 check_against_editor.py [scene.json]

Needs node on PATH. Skips (does not fail) if node or the editor is missing,
so it stays usable in a checkout that has neither.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import sdf_json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
EDITOR = os.path.join(REPO, "sdf_editor.html")
PROBE = os.path.join(HERE, "editor_probe.js")
TOL = 1e-5


def editor_source():
    """The editor lives on main; this branch forked before it landed. Prefer
    the working tree, fall back to git so the check is runnable from here."""
    if os.path.exists(EDITOR):
        return open(EDITOR).read(), EDITOR
    for ref in ("origin/main", "main"):
        r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:sdf_editor.html"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout, f"{ref}:sdf_editor.html"
    return None, None


def extract(src):
    """Lift the evaluator out of the editor. Anchored on declarations rather
    than line numbers so it survives edits above and below them."""
    out = []
    for start, end in (("const PRIMS = {", "const PRIM_KEYS"),
                       ("function smin(", "// The JS twin"),
                       ("function sceneSDF(", "\n}\n")):
        i = src.find(start)
        if i < 0:
            raise SystemExit(f"cannot find {start!r} in the editor")
        j = src.find(end, i + len(start))
        out.append(src[i:j + (len(end) if end == "\n}\n" else 0)])
    out.append("const hits = (n, id) => !n.tg || n.tg.indexOf(id) >= 0;\n")
    return "\n".join(out)


def main(argv):
    scene = argv[1] if len(argv) > 1 else os.path.join(HERE, "owner_joint.json")
    if not shutil.which("node"):
        print("no node on PATH -- skipping")
        return 0
    src, where = editor_source()
    if src is None:
        print("no sdf_editor.html in the tree or on main -- skipping")
        return 0

    harness = os.path.join(HERE, "_editor_harness.js")
    with open(harness, "w") as f:
        f.write(extract(src))
        f.write(open(PROBE).read())
    try:
        ref = subprocess.run(["node", harness, scene], capture_output=True,
                             text=True, check=True)
    finally:
        os.remove(harness)
    rows = np.array([[float(v) for v in ln.split()]
                     for ln in ref.stdout.strip().splitlines()])
    ids = [int(s) for s in ref.stderr.split(":")[-1].strip().split(",")]
    print(f"{os.path.basename(scene)} vs {where}: "
          f"{len(rows)} points, bodies {ids}")

    doc = json.load(open(scene))
    X, Y, Z = rows[:, 0], rows[:, 1], rows[:, 2]
    bad = 0
    for col, bid in enumerate(ids, start=3):
        mine = sdf_json.build(doc, bid, X, Y, Z)
        err = np.abs(mine - rows[:, col])
        agree = np.mean((mine < 0) == (rows[:, col] < 0))
        print(f"  body {bid}: max |err| {err.max():.2e}   "
              f"sign agreement {100 * agree:.3f}%")
        bad += err.max() > TOL or agree < 1.0
    print("\nMATCH" if not bad else f"\nMISMATCH ({bad} body/bodies)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
