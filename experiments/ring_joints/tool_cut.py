"""Check the ring joints the *generator* actually builds.

The linkage is a single solid subtracted from the finished body: cut it out and
what is left is two interlocking pieces. `joint_tool.py` holds the solid and
`flexifish.py` places one per joint; this measures the result.

    python3 tool_cut.py [voxel size] [fixed gap]

Reports, for the default fish: how many pieces the cuts produce, whether
consecutive pieces are linked, the gap between them, and where each one lies.

Linkage is tested by sliding one piece along the body. Pieces that merely nest
come apart at once, so overlap is zero at every displacement. Pieces that are
linked have to pass through each other, so overlap goes positive over a middle
range before they are clear. That signature is the test, and it is the one
thing a shell count cannot tell you -- a fish that falls into loose pieces and
a fish that articulates have exactly the same number of shells.

This deliberately builds the whole fish rather than a window around each joint.
A window is far cheaper and was what this did originally -- but it has to cover
the body's full y and z extent AND the tool's whole reach, and getting either
wrong produces a confident wrong answer. It reported a pass on a truncated body
once, and later reported "one piece" at joints that build correctly. The full
grid cannot be wrong in that way.
"""
import os
import sys

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import flexifish as ff                                   # noqa: E402
import joint_tool                                        # noqa: E402


def linked(A, B, res, axis=0, upto=45.0):
    """Slide B along `axis` and report whether it ever has to pass through A."""
    for k in range(1, int(upto / res)):
        if k >= B.shape[axis]:
            break
        src, dst = [slice(None)] * 3, [slice(None)] * 3
        dst[axis] = slice(k, None)
        src[axis] = slice(None, B.shape[axis] - k)
        shifted = np.zeros_like(B)
        shifted[tuple(dst)] = B[tuple(src)]
        if (A & shifted).any():
            return True
    return False


def main(argv):
    res = float(argv[1]) if len(argv) > 1 else 0.35
    gap = float(argv[2]) if len(argv) > 2 else 0.0
    b = ff.FishBuilder(ff.replace(ff.FishParams(), joint_style="tool",
                                  joint_gap=gap))
    for w in b.warnings:
        print("note:", w)

    x0, x1, y0, y1, z0, z1 = b.bounds()
    xs = np.arange(x0, x1 + res, res, dtype=ff.F32)
    ys = np.arange(y0, y1 + res, res, dtype=ff.F32)
    zs = np.arange(z0, z1 + res, res, dtype=ff.F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    # the body alone: the side fins are separate parts on their own ball
    # joints and would just add shells to count around
    vol = b._split_tool(b.styled(X, Y, Z), X, Y, Z)
    how = "the tool's own" if gap <= 0 else f"{gap} mm"
    print(f"\n{res} mm voxels, clearance = {how}\n")

    lab, n = ndimage.label(vol < 0, ndimage.generate_binary_structure(3, 1))
    sizes = np.array([int((lab == i).sum()) for i in range(1, n + 1)])
    keep = [i + 1 for i in range(n) if sizes[i] > 500]
    ext = {i: np.argwhere(lab == i)[:, 0] for i in keep}
    order = sorted(keep, key=lambda i: ext[i].min())
    want = len(b.joints) + 1
    print(f"{len(order)} body piece(s), expected {want}"
          + (f"   ({n - len(order)} fragment(s) too small to count)"
             if n > len(order) else ""))
    for i in order:
        print(f"  {sizes[i-1] * res ** 3:9.0f} mm3   x "
              f"{ext[i].min() * res + x0:6.1f} ..{ext[i].max() * res + x0:6.1f}")

    bad = len(order) != want or n != len(order)
    print(f"\n{'joint':>6} {'scale':>6} {'gap':>9}  linked")
    for (a, c), j in zip(zip(order, order[1:]), b.joints):
        d = ndimage.distance_transform_edt(~(lab == a), sampling=res)
        g = float(d[lab == c].min())
        ok = linked(lab == a, lab == c, res)
        print(f"{j['xa']:6.0f} {j['s_long']:6.2f} {g:8.2f} mm  {ok}")
        bad |= g < 0.3 or not ok
    print("\nOK" if not bad else "\nFAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
