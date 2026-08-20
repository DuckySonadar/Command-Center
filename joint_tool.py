#!/usr/bin/env python3
"""
joint_tool.py -- the interlocking-ring joint, as a solid that gets subtracted.

The other joint in this generator (ball and socket) is *assembled*: a ball, a
neck, a shell and a relief carve, each sized against the body section so the
shell fits under the skin. That is what makes it fussy -- every term has to be
solved for the local section, and a section too narrow or a segment too short
makes it impossible.

This one is *subtracted*. A single solid, authored in MetaMeld and preserved
verbatim below, is cut out of the finished body; what is left is two pieces
that cannot be pulled apart, because a ring belonging to each passes through
the other.

The parting surface is a thin ellipsoidal *shell*. Subtracting a shell divides
what is inside it from what is outside, so the joint is a ball sitting in a
socket, with the shell's wall as the printed clearance; the rings pass through
that shell and are what stop the ball coming back out. It is a different idea
from the tool it replaced, which was a long cup deliberately larger than any
body it cut. This one has to *contain* the section instead -- if the shell does
not enclose the body, the two halves stay joined and nothing articulates.
HW_REF / H_REF are that containment limit rather than an overshoot.

The tool's frame runs the way the generator does -- x nose to tail, y across,
z up -- so there is no axis swap. Its build plate is its own z = 0, and it
reaches 7.6 mm below that, which is what carries the cut through the fish's
flat belly:

    x = -21.2 ... -10.2 ... 5.6        z = -7.6 .. 22   (plate at z = 0)
                     ^ the joint

Provenance: `LinkageCut.json`, body 0, transcribed exactly -- held to the
document itself at 8.7e-13 mm. `experiments/ring_joints/check_tool_port.py`
holds this to the JS copy in the fish designer.

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
# Exactly the `add`/`cut` nodes of LinkageCut.json that build body 0, in file
# order. The document's ninth node is a disabled vertical plane -- an
# inspection aid for looking at the section in the editor, not geometry -- and
# is deliberately absent: `raw` has no notion of a node being off.
# Order is load-bearing: a shape added after a cut is immune to it, which is
# what lets the big seam cavity be body-scale without destroying the linkage
# that passes through it.
#
#   torus      d = [ring radius, tube radius], axis +z before rotation
#   ellipsoid  d = full diameters
#   plane      the cut removes everything below its own z
#   r          Euler degrees; the shape is rotated Rz*Ry*Rx
#   k          blend radius; 0 is a hard min/max
#
# The document carries a ninth node -- a vertical plane cut, switched off --
# which is an inspection aid for seeing inside the shell while drawing it. It
# is left out rather than carried disabled: `raw` has no notion of a node being
# off, and a switched-off cut is not part of the solid.
TOOL_NODES = [
    ("add", "ellipsoid", 0, (0, 0, 6), (0, -30, 180), (25, 25, 35)),
    ("cut", "ellipsoid", 0, (0, 0, 6), (0, -30, 180), (23, 23, 33)),
    ("cut", "plane", 0, (-6, 0, 0), (0, 62, 180), (0, 0, 0)),
    ("add", "ellipsoid", 4.5, (-14.266247946197, 0, 4), (0, 0, 0), (14, 8.5, 14)),
    ("cut", "torus", 0, (-14, 0, 4), (-86.530223807624, 0, 0), (4.5, 1.5, 0)),
    ("cut", "ellipsoid", 1, (2, 0, 3.816466031276), (91, 13, 180), (30, 26.5, 23)),
    ("add", "torus", 4, (-16.46241654316, 0, 0.100993474262), (-50, 0, 85), (5, 1.75, 0)),
    ("cut", "torus", 2, (-18.269319183782, 0, 1.048664616065), (0, -39, 0), (5, 1.421467019059, 0)),
]

# ---- the tool's own frame ---------------------------------------------
PLATE = 0.0            # its build plate, mapped onto the generator's z = 0
ANCHOR = -10.2         # the joint: the tool-x where the parting surface lands

# How much body a joint consumes, in the tool's units: where removal exceeds
# 2% of its peak, measured by cutting a barrel at the reference section. Nearly
# symmetric about the joint, unlike the old tool's 11 and 35.5, because a shell
# centred on the parting reaches about as far each way.
FOOTPRINT_AHEAD = 10.8
FOOTPRINT_BEHIND = 7.5

# The solid's whole bounding box about the anchor, which is a different and
# larger thing than its footprint. The footprint says where it removes enough
# material to matter for sizing a segment; the box says where it touches
# anything at all. A thin feature standing off the body -- the dorsal fin --
# is severed by a graze that removes almost nothing, so it has to be kept out
# of the box, not merely out of the footprint.
BOX_AHEAD = 11.0
BOX_BEHIND = 15.8

# How close together two of these joints may sit, in tool units.
#
# MIN_SPACING is where a segment still has some of its own body between the
# two shrouds instead of being nothing but shroud. Fewer, longer segments is
# what this joint asks for, and this is the figure the layouts aim at.
#
# SHATTER_SPACING is where it stops working at all: the second cut reaches
# through the first one's interlock and the middle piece comes free. Measured
# at 0.3 mm voxels by cutting a barrel with two of these at a range of
# spacings -- 10 apart still gives three linked pieces, 8 gives three that are
# not linked, plus debris of 0.03 mm3.
MIN_SPACING = FOOTPRINT_AHEAD + FOOTPRINT_BEHIND
SHATTER_SPACING = 10.0

# The section this is scaled against: half-width 10 mm, 20 mm tall. Not the
# largest it can sever -- that is about 11 x 22, where the clearance has
# thinned to 0.6 mm -- but the size at which it still has margin. Sections
# larger than this are not a problem: the tool scales up with them.
HW_REF, H_REF = 10.0, 20.0

# Its separating thickness at the reference section, measured by cutting a
# plain barrel and taking the distance between the two pieces. This is the
# shell's wall, and it is the clearance a joint gets. It scales with the tool,
# so it shrinks along the fish unless `gap` pins it.
THICKNESS = 0.90


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

    The generator runs x nose->tail and so does this solid, so the axes map
    straight through -- unlike the tool this replaced, which ran along its own
    +y and needed a swap.
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
    tx = ((X - xa) / s_long + ANCHOR).astype(F32)
    ty = (Y / s_wide).astype(F32)
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
# The tool was authored in MetaMeld and this is the way back, so it can be
# placed against a baked fish by eye instead of only by `scales_for`.
#
# This got much simpler when the solid changed. The old tool ran along its own
# +y, so the mapping carried an x/y swap -- a reflection, which a rotation
# cannot simply be multiplied by, so each shape's rotation had to be conjugated
# and an ellipsoid's first two diameters exchanged. This solid runs the way the
# generator does. There is no swap left to undo: the rotations come back
# unchanged and only the centres move.
#
# Scaling is uniform only. `tool_sdf` scales each axis on its own, which is a
# thing the editor's shape list cannot say -- a rotated ellipsoid stretched
# along world x is no longer an ellipsoid about its own axes. Export at the
# scale you want, or export at 1 and scale it in MetaMeld.


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
        # out of the tool's frame and into the fish's: the axes already agree,
        # so this is a translation and a scale and nothing else
        w = np.array(p, float) - np.array([ANCHOR, 0.0, PLATE])
        pos = [w[0] * s + xa, w[1] * s, w[2] * s + lift]
        rot = list(r)
        # a torus carries radii and an ellipsoid diameters -- both just scale
        d = [0, 0, 0] if kind == "plane" else [v * s for v in dim]
        nodes.append(dict(t=kind, on=True, op=op, k=k * s, b=0,
                          tg=None if op == "add" else [0], fi=0,
                          # 6 places, not 4: the node centres carry twelve
                          # significant figures, and rounding to 4 moved the
                          # exported solid by 5e-5 mm against tool_sdf
                          p=[round(v, 6) for v in pos],
                          r=[round(v, 6) for v in rot],
                          d=[round(v, 6) for v in d], round=0,
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
        # the solid's own extent with a margin: tool x -21.2..5.6 about an
        # anchor at -10.2, y +/-12, z -7.6..22 off a plate at 0
        lo = [a.at - 14 * s, -15 * s, a.lift - 11 * s]
        hi = [a.at + 19 * s, 15 * s, a.lift + 26 * s]
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
