"""Segment a body by subtracting the owner's joint tool (`tool.json`).

The tool is a single solid authored in the editor: cut it out of a body and
what is left is two interlocking pieces. It is deliberately larger than the
body it cuts, so the outer dimensions do not have to be right -- only the
interlock in the middle sits inside the material -- which is what makes it
scalable in a way a fitted seam cavity is not.

    python3 tool_cut.py [voxel size]

runs it over every joint of the default fish and reports, per joint: how many
pieces the cut produces, whether they are linked, the gap between them, and
the thinnest feature left behind.

Linkage is tested by sliding one piece along the body. Pieces that merely nest
come apart at once, so overlap is zero at every displacement. Pieces that are
linked have to pass through each other, so overlap goes positive over a middle
range before they are clear. That signature is the test.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

import sdf_json

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = json.load(open(os.path.join(HERE, "tool.json")))
TOOL_BODY = 1                 # the only body in that file with any `add` nodes

# The tool's own frame. Its build plate is at z = 8, and its two ring features
# sit at y = 5 and y = 12, so the joint it cuts is centred on their midpoint.
TOOL_PLATE = 8.0
TOOL_ANCHOR = 8.5

# The body the tool was authored against: half-width 20.5 mm, and 38.75 mm of
# height above the plate. Scales are quoted against these.
HW_REF, H_REF = 20.5, 38.75


def tool_sdf(X, Y, Z, xa=0.0, sx=1.0, sy=1.0, sz=1.0):
    """The tool placed at joint `xa`, in the generator's coordinates.

    The generator runs x nose->tail; the tool runs y. Scaling is per-axis, so
    a tall narrow section gets a tall narrow tool. The result is multiplied by
    the smallest scale to keep it an under-estimate of true distance, which is
    what a field being subtracted has to be."""
    ty = ((X - xa) / sy + TOOL_ANCHOR).astype(sdf_json.F32)
    tx = (Y / sx).astype(sdf_json.F32)
    tz = (Z / sz + TOOL_PLATE).astype(sdf_json.F32)
    return sdf_json.build(TOOL, TOOL_BODY, tx, ty, tz) * min(sx, sy, sz)


def scale_for(builder, xa):
    """Scale the tool to the body's section at this joint."""
    top = builder.top_at(xa)
    hw = builder.halfwidth_at(xa, top * 0.45)
    return hw / HW_REF, top / H_REF


def split(body, tool):
    """Cut, then label. Returns piece masks, largest first, and their sizes."""
    solid = (body < 0) & (tool >= 0)
    lab, n = ndimage.label(solid, ndimage.generate_binary_structure(3, 1))
    sizes = np.array([int((lab == i).sum()) for i in range(1, n + 1)])
    order = np.argsort(sizes)[::-1]
    return [lab == (i + 1) for i in order], sizes[order]


def linked(A, B, res, axis=0, upto=45.0):
    """Slide B along `axis` and report whether it ever has to pass through A."""
    for d in np.arange(res * 2, upto, res * 2):
        k = int(round(d / res))
        if k >= B.shape[axis]:
            break
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        dst[axis] = slice(k, None)
        src[axis] = slice(None, B.shape[axis] - k)
        shifted = np.zeros_like(B)
        shifted[tuple(dst)] = B[tuple(src)]
        if (A & shifted).any():
            return True
    return False


def main(argv):
    import importlib.util
    res = float(argv[1]) if len(argv) > 1 else 0.30
    spec = importlib.util.spec_from_file_location(
        "ff", os.path.join(HERE, "flexifish_rings_WIP.py"))
    ff = importlib.util.module_from_spec(spec)
    sys.modules["ff"] = ff
    spec.loader.exec_module(ff)

    b = ff.FishBuilder(ff.FishParams())
    print(f"tool.json at {res} mm voxels\n")
    print(f"{'joint':>6} {'section':>14} {'scale':>16} {'pieces':>7} "
          f"{'linked':>7} {'gap':>8} {'thinnest':>9}")
    bad = 0
    for j in b.joints:
        xa = j["xa"]
        sx, sz = scale_for(b, xa)
        sy = sz
        pad = 30 * max(sy, 1.0)
        xs = np.arange(xa - pad, xa + pad, res)
        ys = np.arange(-26, 26, res)
        zs = np.arange(0, 34, res)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        F = b.body_field(X.astype(ff.F32), Y.astype(ff.F32), Z.astype(ff.F32),
                         side_fins=False)
        T = tool_sdf(X, Y, Z, xa=xa, sx=sx, sy=sy, sz=sz)
        pieces, sizes = split(F, T)
        big = [p for p, s in zip(pieces, sizes) if s > 500]
        gap = thin = float("nan")
        ok = False
        if len(big) >= 2:
            A, B = big[0], big[1]
            ok = linked(A, B, res)
            gap = ndimage.distance_transform_edt(~A, sampling=res)[B].min()
            near = np.abs(X - xa) < 14 * max(sy, 1.0)
            thin = min(2 * ndimage.distance_transform_edt(
                p & near, sampling=res).max() for p in (A, B))
        top = b.top_at(xa)
        hw = b.halfwidth_at(xa, top * 0.45)
        print(f"{xa:6.0f} {hw:6.1f} x {top:5.1f} "
              f"{sx:5.2f}/{sy:4.2f}/{sz:4.2f} {len(big):7d} "
              f"{str(ok):>7} {gap:7.2f}mm {thin:8.1f}mm")
        bad += (len(big) != 2) or not ok
    print("\nall joints split into two linked pieces" if not bad
          else f"\n{bad} joint(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
