"""Range of motion: yaw the rear segment and find where it hits the front.

The dome and the cup are concentric spheres, so the joint is a ball joint and
its motion is a yaw about the dome centre. Rotate the rear segment's field
about that axis and count voxels where the two segments are both solid.

    python3 check_swing.py [joint index] [voxel size]

What a healthy joint looks like: zero overlap through the design swing, and
the first contact inside the dome -- the rings meeting each other, which is
the end stop the design intends. Contact out on the body face instead means
the seam wedge is closing before the linkage does.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "flexifish_rings_WIP", os.path.join(HERE, "flexifish_rings_WIP.py"))
ff = importlib.util.module_from_spec(_spec)
sys.modules["flexifish_rings_WIP"] = ff
_spec.loader.exec_module(ff)
F32 = ff.F32


def main(argv):
    ji = int(argv[1]) if len(argv) > 1 else 0
    res = float(argv[2]) if len(argv) > 2 else 0.25
    b = ff.FishBuilder(ff.FishParams())
    j = b.joints[ji]
    xa, zc, R, rt = j["xa"], j["zc"], j["R"], j["rt"]

    span = j["off"] / 2 + R + rt + 4.0
    xs = np.arange(xa - span, xa + span + res, res, dtype=F32)
    ys = np.arange(-(R + rt + 6.0), R + rt + 6.0 + res, res, dtype=F32)
    zs = np.arange(-1.0, zc + R + rt + 4.0 + res, res, dtype=F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    front = b.segment(ji, X, Y, Z) < 0

    print(f"joint {ji} at x={xa:.1f}: design swing {j['swing']:.1f} deg "
          f"(+/-{j['swing']/2:.1f}), seam wedge opens "
          f"{2*np.degrees(np.arctan(j['tanb'])):.1f} deg, "
          f"relief swept +/-{j['swing']/2 + 1:.1f} deg")
    print(f"yawing the rear segment about ({xa:.1f}, 0, {zc:.2f}) at {res} mm\n")
    print(f"{'yaw':>5} {'overlap':>9}  contact")
    limit = None
    for deg in (0, 2, 4, 6, 8, 10, 12, 14, 16, 20):
        th = np.deg2rad(-deg)                    # inverse-rotate the samples
        c, s = np.cos(th), np.sin(th)
        dx = X - xa
        Xr = (xa + c * dx - s * Y).astype(F32)
        Yr = (s * dx + c * Y).astype(F32)
        ov = front & (b.segment(ji + 1, Xr, Yr, Z) < 0)
        n = int(ov.sum())
        note = ""
        if n:
            if limit is None:
                limit = deg
            idx = np.argwhere(ov)
            d = np.sqrt((xs[idx[:, 0]] - xa) ** 2 + ys[idx[:, 1]] ** 2
                        + (zs[idx[:, 2]] - zc) ** 2)
            inside = float((d < j["dome"]).mean())
            note = (f"{100*inside:3.0f}% inside the dome -- "
                    + ("rings (the intended stop)" if inside > 0.5
                       else "BODY FACE (the seam is binding first)"))
        print(f"{deg:5d} {n:9d}  {note}")

    reach = j["swing"] / 2
    got = (limit if limit is not None else 999)
    print(f"\nfirst contact at {got if limit is not None else '>20'} deg; "
          f"the design asks for +/-{reach:.1f}")
    ok = got > reach
    print("PASS" if ok else "FAIL: binds inside the design swing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
