"""Island check: does any part of the joint start in mid-air?

Walk the coupon layer by layer. A connected component of one layer that
overlaps nothing in the layer below is an island -- material the printer has
to start in free space. This is the check the README kept deferring as "the
overhang audit", and it is what caught the tilted ring being wrong way up:
with the ring's axis tilted toward +x, the protruding arc is the ring's *low*
half and its bottom sits out past the seam with nothing under it.

    python3 check_print.py [joint index] [voxel size]

Volumes are reported because size is what decides whether an island matters.
A fraction of a cubic millimetre is one layer of droop that the next layers
absorb; several cubic millimetres is a feature that needs support.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "flexifish_rings_WIP", os.path.join(HERE, "flexifish_rings_WIP.py"))
ff = importlib.util.module_from_spec(_spec)
sys.modules["flexifish_rings_WIP"] = ff
_spec.loader.exec_module(ff)
F32 = ff.F32

# anything smaller than this is a droop the next layers close over, not a
# feature the slicer has to support
OK_MM3 = 1.0


def main(argv):
    ji = int(argv[1]) if len(argv) > 1 else 0
    res = float(argv[2]) if len(argv) > 2 else 0.20
    b = ff.FishBuilder(ff.FishParams())
    j = b.joints[ji]

    reach = j["off"] / 2 + j["R"] + j["rt"]
    x0, x1 = j["xa"] - reach - 2.0, j["xa"] + reach + 2.0
    y0, y1 = -(j["R"] + j["rt"] + 2.5), j["R"] + j["rt"] + 2.5
    z1 = j["zc"] + j["R"] + j["rt"] + 2.5
    xs = np.arange(x0 - 1, x1 + 1 + res, res, dtype=F32)
    ys = np.arange(y0 - 1, y1 + 1 + res, res, dtype=F32)
    # start inside the first layer: at z = 0 exactly the plate cut makes the
    # field zero, so layer 0 would read as empty and layer 1 as an island
    zs = np.arange(res / 2, z1 + res, res, dtype=F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    F = b.body_field(X, Y, Z, side_fins=False)
    sf = b.segment(ji, X, Y, Z, F=F)
    sr = b.segment(ji + 1, X, Y, Z, F=F)
    vol = np.maximum(np.minimum(sf, sr),
                     ff.sd_box(X, Y, Z, x0, x1, y0, y1, -5.0, z1))
    solid = vol < 0

    print(f"joint {ji} at x={j['xa']:.1f}, res={res} mm, "
          f"R={j['R']:.2f} tube={j['rt']:.2f} dome={j['dome']:.2f}")
    print(f"{'z':>7} {'volume':>9}  segment  where")
    st = ndimage.generate_binary_structure(2, 2)
    worst = 0.0
    n_isl = 0
    for k in range(1, len(zs)):
        lab, n = ndimage.label(solid[:, :, k], st)
        if not n:
            continue
        below = solid[:, :, k - 1]
        for c in range(1, n + 1):
            m = lab == c
            if (m & below).any():
                continue
            n_isl += 1
            mm3 = float(m.sum()) * res ** 3
            worst = max(worst, mm3)
            idx = np.argwhere(m)
            who = "front" if (sf[:, :, k][m] < 0).mean() > 0.5 else "rear"
            print(f"{zs[k]:7.2f} {mm3:7.2f}mm3  {who:7s}  "
                  f"x {xs[idx[:,0]].min():6.2f}..{xs[idx[:,0]].max():6.2f}  "
                  f"y {ys[idx[:,1]].min():6.2f}..{ys[idx[:,1]].max():6.2f}")
    if not n_isl:
        print("   (none)")
    print(f"\n{n_isl} island(s), largest {worst:.2f} mm3")
    ok = worst <= OK_MM3
    print("PASS" if ok else f"FAIL: an island over {OK_MM3} mm3 needs support")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
