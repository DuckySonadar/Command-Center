"""Derive joint_tool.py's frame constants from the solid itself.

The originals were measured by hand for the first tool. When the solid is
re-authored in the editor those numbers do not carry over -- the tool is
placed by `PLATE`, scaled against `HW_REF`/`H_REF`, spaced by `FOOTPRINT_*`
and gives its clearance from `THICKNESS`, and every one of them describes a
particular solid. Reading them off the wrong one is silent: the tool lands at
the wrong height, or is sized for a section it was never drawn around, and
what comes out is a fish in loose pieces.

    python3 derive_tool.py [voxel size]

Everything here is measured against a plain elliptical barrel rather than a
fish, so the numbers describe the tool and nothing else.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))    # the repo root
sys.path.insert(0, HERE)
import joint_tool as jt                                       # noqa: E402

from tool_cut import linked                                   # noqa: E402


def barrel(X, Y, Z, hw, h):
    """An elliptical barrel along x, half-width `hw`, `h` tall, on z = 0."""
    return np.maximum(np.sqrt((Y / hw) ** 2 + (Z / h) ** 2) - 1.0, -Z)


def pieces(field, res, min_mm3=2.0):
    """Label the solid; return (volumes, masks) biggest first, dropping dust."""
    lab, n = ndimage.label(field < 0)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        v = m.sum() * res ** 3
        if v >= min_mm3:
            out.append((v, m))
    out.sort(key=lambda t: -t[0])
    return out


def separation(a, b, res, grid):
    """Smallest gap between two labelled pieces, in mm."""
    d = ndimage.distance_transform_edt(~a, sampling=(res, res, res))
    return float(d[b].min())


def cut_barrel(res, hw, h, xa=0.0, span=90.0, second=None):
    x = np.arange(-span, span + res, res, dtype=np.float32)
    y = np.arange(-hw - 12, hw + 12 + res, res, dtype=np.float32)
    z = np.arange(-2, h + 14 + res, res, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    body = barrel(X, Y, Z, hw, h)
    s_w, s_t = hw / jt.HW_REF, h / jt.H_REF
    body = np.maximum(body, -jt.tool_sdf(X, Y, Z, xa, s_w, s_t, s_t))
    if second is not None:
        body = np.maximum(body, -jt.tool_sdf(X, Y, Z, second, s_w, s_t, s_t))
    return body, (X, Y, Z)


def main(argv):
    res = float(argv[0]) if argv else 0.5
    print(f"voxel {res} mm\n")

    # ---- the solid's own frame -------------------------------------
    g = np.mgrid[-45:45:res, -20:80:res, -20:80:res].astype(np.float32)
    ins = jt.raw(g[0], g[1], g[2]) < 0
    zmin, zmax = g[2][ins].min(), g[2][ins].max()
    ymin, ymax = g[1][ins].min(), g[1][ins].max()
    xhw = float(np.abs(g[0][ins]).max())
    print("the solid itself")
    print(f"  x  +/-{xhw:6.2f}")
    print(f"  y  {ymin:6.2f} .. {ymax:6.2f}")
    print(f"  z  {zmin:6.2f} .. {zmax:6.2f}")
    print(f"  PLATE   = {zmin:.2f}   (its lowest material; the fish's z=0)")
    print(f"  current = {jt.PLATE:.2f}\n")

    # ---- what section it severs cleanly ----------------------------
    # The tool overshoots on purpose, so it severs any section smaller than
    # the one it was drawn around -- "does it work" is not the question. The
    # design section is where it *stops* working: grow the barrel until the
    # cut no longer reaches across it.
    print("cutting a barrel at scale 1.0 -- growing until it stops severing")
    print("   half-w  height   pieces  linked   gap mm")

    def try_section(hw, h):
        body, _ = cut_barrel(res, hw, h)
        ps = pieces(body, res)
        if len(ps) != 2:
            return None, len(ps)
        if not linked(ps[0][1], ps[1][1], res):
            return None, len(ps)
        return separation(ps[0][1], ps[1][1], res, None), len(ps)

    hw_max, h_probe = None, 34.0
    for hw in (16.0, 20.5, 25.0, 30.0, 35.0, 40.0, 46.0):
        gap, n = try_section(hw, h_probe)
        print(f"   {hw:6.1f}  {h_probe:6.2f}   {n:5d}   "
              f"{str(gap is not None):6s}  {'' if gap is None else f'{gap:5.2f}'}")
        if gap is None:
            break
        hw_max = hw
    h_max = None
    for h in (26.0, 30.0, 34.0, 38.75, 44.0, 50.0, 58.0):
        gap, n = try_section(hw_max or 20.5, h)
        print(f"   {(hw_max or 20.5):6.1f}  {h:6.2f}   {n:5d}   "
              f"{str(gap is not None):6s}  {'' if gap is None else f'{gap:5.2f}'}")
        if gap is None:
            break
        h_max, thick = h, gap
    if hw_max is None or h_max is None:
        print("\n  no section severed into two linked pieces")
        return 1
    print(f"\n  widest section it still severs: {hw_max:.1f} half-width,"
          f" {h_max:.2f} tall")
    print(f"  HW_REF / H_REF = {hw_max:.1f} / {h_max:.2f}"
          f"   (current {jt.HW_REF} / {jt.H_REF})")
    print(f"  THICKNESS      = {thick:.2f}   (current {jt.THICKNESS})")

    hw, h = hw_max, h_max

    # ---- footprint: where along x it actually removes material -----
    body, (X, _, _) = cut_barrel(res, hw, h, xa=0.0)
    whole = barrel(*np.meshgrid(
        np.arange(-90, 90 + res, res, dtype=np.float32),
        np.arange(-hw - 12, hw + 12 + res, res, dtype=np.float32),
        np.arange(-2, h + 14 + res, res, dtype=np.float32),
        indexing="ij"), hw, h)
    removed = (whole < 0) & ~(body < 0)
    per_x = removed.sum(axis=(1, 2)) * res ** 3
    xs = np.arange(-90, 90 + res, res)
    live = per_x > 0.02 * per_x.max()
    ahead, behind = -xs[live].min(), xs[live].max()
    print(f"\nfootprint (>2% of peak removal), anchor at x = 0")
    print(f"  AHEAD  = {ahead:5.1f}   (current {jt.FOOTPRINT_AHEAD})")
    print(f"  BEHIND = {behind:5.1f}   (current {jt.FOOTPRINT_BEHIND})")
    print(f"  MIN_SPACING = {ahead + behind:.1f}"
          f"   (current {jt.MIN_SPACING})")

    # ---- two tools: where does the middle segment fall apart? ------
    print("\ntwo tools on one barrel")
    print("   spacing  pieces  smallest mm3")
    shatter = None
    for sp in (10, 12, 14, 18, 24, 30):
        body, _ = cut_barrel(res, hw, h, xa=-sp / 2, second=sp / 2)
        ps = pieces(body, res, min_mm3=0.0)
        big = [v for v, _ in ps if v >= 2.0]
        print(f"   {sp:7.0f}  {len(big):5d}   {min(v for v, _ in ps):10.1f}")
        if len(big) != 3 and shatter is None:
            shatter = sp
    print(f"  SHATTER_SPACING = "
          f"{'none found in range' if shatter is None else shatter}"
          f"   (current {jt.SHATTER_SPACING})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
