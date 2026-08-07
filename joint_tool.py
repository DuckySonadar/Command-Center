#!/usr/bin/env python3
"""
joint_tool.py -- the interlocking-ring joint, as a solid that gets subtracted.

The other joint in this generator (ball and socket) is *assembled*: a ball, a
neck, a shell and a relief carve, each sized against the body section so the
shell fits under the skin. That is what makes it fussy -- every term has to be
solved for the local section, and a section too narrow or a segment too short
makes it impossible.

This one is *subtracted*. A single solid, authored in the repo's
`sdf_editor.html` and preserved verbatim below, is cut out of the finished
body; what is left is two pieces that cannot be pulled apart, because a ring
belonging to each passes through the other. The solid is deliberately larger
than any body it will cut -- only the interlock in the middle is ever inside
the material, and the rest of it hangs outside in free space. That overshoot is
the point: the outer dimensions do not have to be right, so the whole thing can
be scaled to the local section without solving anything.

The tool's own frame (its build plate is z = 8, and it runs along +y):

    y = -2.5 ......... 8.5 ......... 44        z = 8 (plate) .. 49.75
                       ^ the joint

so it is lopsided about the joint it cuts: 11 mm ahead, 35.5 mm behind. Most of
that tail is the two walls of a cup that wraps the rear piece's nose from the
outside, well clear of the body; the part that actually removes material stops
around 22 mm behind (see FOOTPRINT_BEHIND).

Provenance: `experiments/ring_joints/tool.json`, body 1. The scene is embedded
here rather than loaded so that flexifish.py stays a single-file generator with
no data files beside it. `experiments/ring_joints/check_against_editor.py`
holds the maths below to the editor's own `sceneSDF` -- currently agreeing to
1.9e-7 mm -- so do not change it without re-running that.

Run as a script, it goes the other way and writes the solid back out as a
MetaMeld scene, placed where a joint would sit:

    python3 joint_tool.py --at 62.5 --scale 0.8 --out cutter.json --check
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


# ======================================================================
# The scene
# ======================================================================
# Exactly the `add`/`cut` nodes of tool.json that build body 1, in file order.
# Order is load-bearing: a shape added after a cut is immune to it, which is
# what lets the big seam cavity be body-scale without destroying the linkage
# that passes through it.
#
#   torus      d = [ring radius, tube radius], axis +z before rotation
#   ellipsoid  d = full diameters
#   plane      the cut removes everything below its own z
#   r          Euler degrees; the shape is rotated Rz*Ry*Rx
#   k          blend radius; 0 is a hard min/max
TOOL_NODES = [
    ("add", "ellipsoid", 2.75, (0, 29, 27), (-42, 0, 0), (54.5, 45.5, 82.5)),
    ("cut", "ellipsoid", 5.00, (0, 47.5, 29), (-51, 0, 0), (46, 59, 78.5)),
    ("cut", "ellipsoid", 4.25, (0, 25, 9), (0, 0, 0), (15.5, 22, 14)),
    ("add", "torus", 1.50, (0, 5, 15.5), (30, 0, 0), (8, 5.75, 0)),
    ("cut", "torus", 2.00, (0, 12, 19), (0, 90, 0), (8, 3, 0)),
    ("cut", "torus", 3.50, (0, 4.5, 15), (30, 0, 0), (8.5, 2.75, 0)),
    ("cut", "ellipsoid", 1.50, (0, -6, 11), (-58, 0, 0), (30, 18, 14)),
    ("cut", "plane", 2.00, (0, 0, 8), (0, 0, 0), (0, 0, 0)),
]

# ---- the tool's own frame ---------------------------------------------
PLATE = 8.0            # its build plate, mapped onto the generator's z = 0
ANCHOR = 8.5           # the joint, midway between its two ring features

# How much body a joint consumes, in the tool's units. The bounding box is 11
# ahead of the anchor and 35.75 behind, but most of that tail is cup wall in
# free space: cutting a real body section, removal starts at the tool's very
# front and has fallen to a few percent of the section by 15 behind, trailing
# off to nothing around 22 where the walls only grazes the shoulder.
FOOTPRINT_AHEAD = 11.0
FOOTPRINT_BEHIND = 15.0

# How close together two of these joints may sit, in tool units.
#
# MIN_SPACING is where a segment still has some of its own body between the
# two shrouds instead of being nothing but shroud. Fewer, longer segments is
# what this joint asks for, and this is the figure the layouts aim at.
#
# SHATTER_SPACING is where it stops working at all: the second tool cuts
# through the first one's interlock and the body falls apart. Measured at
# 0.4 mm voxels by cutting one fish with two tools at a range of spacings and
# labelling the result -- 12 apart gives three linked pieces, 10 gives five,
# two of them debris under 2 mm3.
MIN_SPACING = FOOTPRINT_AHEAD + FOOTPRINT_BEHIND
SHATTER_SPACING = 12.0

# The body it was authored against: half-width 20.5 mm, and 38.75 mm of height
# above the plate. Scales are quoted against these.
HW_REF, H_REF = 20.5, 38.75

# Its separating thickness at scale 1.0, measured by cutting a plain barrel and
# taking the distance between the two pieces. Left alone, this is the clearance
# a joint gets, and it scales with the tool -- so it shrinks along the fish
# unless `gap` pins it. Nothing else in the build adds to it.
THICKNESS = 1.20


# ======================================================================
# Evaluating it
# ======================================================================
def _rot(x, y, z, deg):
    """World -> local. The editor rotates a shape by Rz*Ry*Rx, so the inverse
    undoes Z first, then Y, then X."""
    for ax, a in zip((2, 1, 0), np.deg2rad(deg)[::-1]):
        if a == 0:
            continue
        c, s = np.cos(-a), np.sin(-a)          # inverse
        if ax == 0:
            y, z = c * y - s * z, s * y + c * z
        elif ax == 1:
            z, x = c * z - s * x, s * z + c * x
        else:
            x, y = c * x - s * y, s * x + c * y
    return x, y, z


def _smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _node(kind, p, r, d, X, Y, Z):
    x, y, z = _rot(X - p[0], Y - p[1], Z - p[2], np.asarray(r, float))
    if kind == "torus":
        q = np.sqrt(x * x + y * y) - d[0]
        return np.sqrt(q * q + z * z) - d[1]
    if kind == "ellipsoid":
        a, b, c = d[0] / 2, d[1] / 2, d[2] / 2
        k0 = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
        k1 = np.sqrt((x / a ** 2) ** 2 + (y / b ** 2) ** 2 + (z / c ** 2) ** 2)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
    if kind == "plane":                        # the region the cut removes
        # the editor evaluates pPlane on the *rotated* point (`q.z`), so a
        # tilted plane tilts. Reading world Z here silently flattened every
        # rotated plane into a horizontal one at z = p[2].
        return z
    raise ValueError(kind)


def raw(X, Y, Z):
    """The tool in its own coordinates, unscaled and unplaced."""
    d = None
    for op, kind, k, p, r, dim in TOOL_NODES:
        f = _node(kind, p, r, dim, X, Y, Z)
        if op == "add":
            d = f if d is None else _smin(d, f, k)
        else:
            d = -_smin(-d, f, k)               # smax(d, -f, k)
    return d


def tool_sdf(X, Y, Z, xa=0.0, s_wide=1.0, s_long=1.0, s_tall=1.0, gap=None,
             lift=0.0):
    """The tool placed at joint `xa`, in the generator's coordinates.

    The generator runs x nose->tail and the tool runs y, so the axes swap.
    Scaling is per-axis: a tall narrow section gets a tall narrow tool. The
    result is multiplied by the smallest scale, which keeps it an
    under-estimate of true distance -- what a field being subtracted has to be
    if the blends around it are to behave.

    `gap` overrides the clearance. Without it the joint gets the tool's own
    thickness times the scale, so the clearance shrinks along the fish along
    with everything else. With it, the tool is eroded (or dilated) by half the
    difference after scaling, so every joint gets the same gap whatever the
    section is.

    `lift` raises the tool off the build plate. Its own plate lands on z = 0
    by default, which is where a joint wants to be on a fish that prints
    belly-down; lifting it is for looking at what the cut is doing, not
    something a finished print usually wants."""
    tx = (Y / s_wide).astype(F32)
    ty = ((X - xa) / s_long + ANCHOR).astype(F32)
    tz = ((Z - lift) / s_tall + PLATE).astype(F32)
    s = min(s_wide, s_long, s_tall)
    d = raw(tx, ty, tz) * s
    if gap is not None:
        d = d + (THICKNESS * s - gap) / 2.0
    return d.astype(F32)


def scales_for(builder, xa):
    """Scale the tool to the body's section at joint `xa`.

    Returns (wide, long, tall). Fore-aft follows the body's *height*, not its
    length: it is the section that decides how big an interlock will fit, and
    a joint whose rings shrank in two axes but not the third would not be the
    same joint."""
    top = builder.top_at(xa)
    hw = builder.halfwidth_at(xa, top * 0.45)
    return hw / HW_REF, top / H_REF, top / H_REF


def footprint(s_long):
    """(ahead, behind) in mm: how much of the body a joint at this scale
    consumes. Two joints closer together than ahead+behind cut into each
    other's interlock and the segment between them comes out in pieces."""
    return FOOTPRINT_AHEAD * s_long, FOOTPRINT_BEHIND * s_long


# ======================================================================
# Handing it to MetaMeld
# ======================================================================
# The tool was authored in the editor and this is the way back, so it can be
# placed against a baked fish by eye instead of only by `scales_for`. The two
# frames differ: the tool runs along its own +y with its plate at z = PLATE,
# the generator runs x nose->tail with the plate at z = 0. `tool_sdf` maps one
# to the other with t = S.w/s + c, S being the x/y swap, so the scene has to
# carry S the other way.
#
# S is a reflection, so a shape's rotation cannot simply be multiplied by it --
# S.R is not a rotation. What works is the conjugate S.R.S, which is, paid for
# by evaluating the shape in a locally swapped frame. Every primitive here
# survives that swap: a torus and a plane are about their local z and do not
# notice, an ellipsoid does, and gets its first two diameters exchanged.
#
# Scaling is uniform only. `tool_sdf` scales each axis on its own, which is a
# thing the editor's shape list cannot say -- a rotated ellipsoid stretched
# along world x is no longer an ellipsoid about its own axes. Export at the
# scale you want, or export at 1 and scale it in MetaMeld.
_SWAP = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])


def _euler_to_m(r):
    """Rz*Ry*Rx, degrees in -- the editor's convention."""
    a, b, c = np.deg2rad(r)
    ca, sa, cb, sb, cc, sc = (np.cos(a), np.sin(a), np.cos(b),
                              np.sin(b), np.cos(c), np.sin(c))
    return np.array([
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb,     cb * sa,                cb * ca]])


def _m_to_euler(m):
    b = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
    if abs(np.cos(b)) > 1e-7:
        a = np.arctan2(m[2, 1], m[2, 2])
        c = np.arctan2(m[1, 0], m[0, 0])
    else:                                   # pitch at +/-90: roll takes it all
        a, c = np.arctan2(-m[1, 2], m[1, 1]), 0.0
    return [float(np.rad2deg(v)) for v in (a, b, c)]


def to_metameld(xa=0.0, scale=1.0, lift=0.0, name="Cutter"):
    """The tool as an editor document, in the generator's coordinates.

    Placed at joint `xa` on the fish's long axis, its plate `lift` above
    z = 0, at a uniform `scale` -- the same solid that
    `tool_sdf(X, Y, Z, xa, s, s, s, lift=lift)` cuts.
    The cuts target the cutter's own body, so importing this beside a fish
    carves nothing: it arrives as a solid to be looked at and moved."""
    s = float(scale)
    nodes = []
    for op, kind, k, p, r, dim in TOOL_NODES:
        # centre: out of the tool's frame, then swapped into the fish's
        w = _SWAP @ (np.array(p, float) - np.array([0.0, ANCHOR, PLATE]))
        pos = [w[0] * s + xa, w[1] * s, w[2] * s + lift]
        rot = _m_to_euler(_SWAP @ _euler_to_m(r) @ _SWAP)
        # a torus carries radii and an ellipsoid diameters -- both just scale.
        # The local x/y swap the conjugation leaves behind is invisible to a
        # torus (it is about its local z) but not to an ellipsoid.
        d = [dim[1], dim[0], dim[2]] if kind == "ellipsoid" else list(dim)
        d = [0, 0, 0] if kind == "plane" else [v * s for v in d]
        nodes.append(dict(t=kind, on=True, op=op, k=k * s, b=0,
                          tg=None if op == "add" else [0], fi=0,
                          p=[round(v, 4) for v in pos],
                          r=[round(v, 4) for v in rot],
                          d=[round(v, 4) for v in d], round=0,
                          mx=False, my=False, mz=False))
    return {"version": 2, "bodies": [{"id": 0, "name": name, "on": True}],
            "nodes": nodes}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Write the joint cutter as a MetaMeld scene.")
    ap.add_argument("--at", type=float, default=0.0,
                    help="joint position along the fish, mm (default 0)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="uniform scale (default 1; scales_for's is per-axis "
                         "and cannot be written as a scene)")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="raise it off the build plate, mm (default 0)")
    ap.add_argument("--name", default="Cutter", help="the body's name")
    ap.add_argument("--out", default="cutter.json")
    ap.add_argument("--check", action="store_true",
                    help="re-read the file as the editor does and hold it to "
                         "tool_sdf")
    a = ap.parse_args()
    doc = to_metameld(a.at, a.scale, a.lift, a.name)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"{a.out}: {len(doc['nodes'])} nodes, joint at x={a.at:g}, "
          f"scale {a.scale:g}, lift {a.lift:g}")

    if a.check:
        import os
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "experiments", "ring_joints"))
        import sdf_json                                    # the editor's twin

        s = a.scale
        rng = np.random.default_rng(7)
        lo = [a.at - 20 * s, -35 * s, a.lift - 6 * s]
        hi = [a.at + 45 * s, 35 * s, a.lift + 50 * s]
        P = rng.uniform(lo, hi, size=(20000, 3))
        X, Y, Z = (P[:, i].astype(F32) for i in range(3))
        mine = tool_sdf(X, Y, Z, a.at, s, s, s, lift=a.lift)
        theirs = sdf_json.build(json.loads(json.dumps(doc)), 0, X, Y, Z)
        err = float(np.max(np.abs(mine - theirs)))
        agree = float(np.mean((mine < 0) == (theirs < 0)) * 100)
        inside = int((mine < 0).sum())
        print(f"vs tool_sdf over {len(P)} points ({inside} inside): "
              f"max |err| {err:.3g} mm, same sign {agree:.4f}%")
        raise SystemExit(0 if err < 1e-3 else 1)
