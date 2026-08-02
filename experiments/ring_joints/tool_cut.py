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

# How far the tool reaches either side of the joint, in its own units. It
# spans y -2.5 .. 44 about an anchor at 8.5, so it is very lopsided: a test
# window has to cover +35.5 behind the joint or the cut lands incomplete and
# the pieces stay joined through the part that was never cut.
TOOL_AHEAD, TOOL_BEHIND = 11.0, 35.5

# The body the tool was authored against: half-width 20.5 mm, and 38.75 mm of
# height above the plate. Scales are quoted against these.
HW_REF, H_REF = 20.5, 38.75

# The tool's own separating thickness at scale 1.0, measured by cutting a
# plain barrel and taking the distance between the two pieces. This is the
# clearance the joint gets, and it scales with the tool: 1.20 mm at the head
# joint, 0.60 mm at the tail. Nothing else in the build adds to it.
TOOL_THICKNESS = 1.20


def tool_sdf(X, Y, Z, xa=0.0, sx=1.0, sy=1.0, sz=1.0, gap=None):
    """The tool placed at joint `xa`, in the generator's coordinates.

    The generator runs x nose->tail; the tool runs y. Scaling is per-axis, so
    a tall narrow section gets a tall narrow tool. The result is multiplied by
    the smallest scale to keep it an under-estimate of true distance, which is
    what a field being subtracted has to be.

    `gap` overrides the clearance. Without it the joint gets the tool's own
    thickness times the scale, so the clearance shrinks along the fish along
    with everything else -- 1.20 mm at the head, 0.60 mm at the tail. With it,
    the tool is eroded (or dilated) by half the difference after scaling, so
    every joint gets the same gap whatever the section is."""
    ty = ((X - xa) / sy + TOOL_ANCHOR).astype(sdf_json.F32)
    tx = (Y / sx).astype(sdf_json.F32)
    tz = (Z / sz + TOOL_PLATE).astype(sdf_json.F32)
    d = sdf_json.build(TOOL, TOOL_BODY, tx, ty, tz) * min(sx, sy, sz)
    if gap is not None:
        d = d + (TOOL_THICKNESS * min(sx, sy, sz) - gap) / 2.0
    return d


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
    """Cut every joint of the default fish and measure the result.

    Deliberately builds the whole fish rather than a window around each joint.
    A window is far cheaper and was what this did originally -- but it has to
    cover the body's full y and z extent AND the tool's whole reach, and
    getting either wrong produces a confident wrong answer. It reported a pass
    on a truncated body once, and later reported "one piece" at joints that
    build correctly. The full grid cannot be wrong in that way.
    """
    import importlib.util
    res = float(argv[1]) if len(argv) > 1 else 0.30
    gap = float(argv[2]) if len(argv) > 2 else None
    spec = importlib.util.spec_from_file_location(
        "ff", os.path.join(HERE, "flexifish_rings_WIP.py"))
    ff = importlib.util.module_from_spec(spec)
    sys.modules["ff"] = ff
    spec.loader.exec_module(ff)

    b = ff.FishBuilder(ff.FishParams())
    x0, x1, y0, y1, z0, z1 = b.bounds()
    xs = np.arange(x0, x1 + res, res, dtype=ff.F32)
    ys = np.arange(y0, y1 + res, res, dtype=ff.F32)
    zs = np.arange(z0, z1 + res, res, dtype=ff.F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = b.body_field(X, Y, Z, side_fins=False)
    how = "the tool's own" if gap is None else f"{gap} mm"
    print(f"tool.json at {res} mm voxels, clearance = {how}\n")
    for j in b.joints:
        sx, sz = scale_for(b, j["xa"])
        vol = np.maximum(vol, -tool_sdf(X, Y, Z, xa=j["xa"], sx=sx, sy=sz,
                                        sz=sz, gap=gap).astype(ff.F32))

    lab, n = ndimage.label(vol < 0, ndimage.generate_binary_structure(3, 1))
    sizes = np.array([int((lab == i).sum()) for i in range(1, n + 1)])
    keep = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] > 500]
    ext = {i: np.argwhere(lab == i)[:, 0] for i in keep}
    order = sorted(keep, key=lambda i: ext[i].min())
    print(f"{len(order)} body piece(s), expected {len(b.joints) + 1}")
    for i in order:
        print(f"  {sizes[i-1]:9d} vox   x {ext[i].min()*res + x0:6.1f}"
              f" ..{ext[i].max()*res + x0:6.1f}")
    print(f"\n{'joint':>6} {'scale':>6} {'gap':>8}")
    bad = len(order) != len(b.joints) + 1
    for (a, c), j in zip(zip(order, order[1:]), b.joints):
        d = ndimage.distance_transform_edt(~(lab == a), sampling=res)
        g = d[lab == c].min()
        sx, sz = scale_for(b, j["xa"])
        print(f"{j['xa']:6.0f} {min(sx, sz):6.2f} {g:7.2f} mm")
        bad |= g < 0.3
    print("\nOK" if not bad else "\nFAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
