"""Evaluate a scene saved by the repo's `sdf_editor.html` as an SDF.

A twin of the editor's own `sceneSDF`, so a model built there can be measured
and meshed here directly rather than transcribed. `check_against_editor.py`
holds the two to each other; do not change the maths in this file without
running it.

Conventions, all taken from the editor rather than guessed:
  torus      d = [ring radius, tube radius], axis +z before rotation
  ellipsoid  d = full diameters
  plane      cut keeps everything above its own z
  r          Euler degrees; the shape is rotated Rz*Ry*Rx, so world -> local
             undoes Z, then Y, then X
  k          blend radius; 0 is a hard min/max
  add        belongs to body `b`;  cut applies to bodies in `tg`
  tg = null  cuts every body;  tg = [] cuts none
Bodies never blend into each other -- they meet in a plain min.
"""
import numpy as np

F32 = np.float32


def rot(p, deg):
    """World -> local. The editor rotates a shape by Rz*Ry*Rx, so the inverse
    undoes Z first, then Y, then X. Order only matters when two axes are both
    non-zero, which is why building the joint file the wrong way round still
    gave the right answer -- every rotation in it is single-axis."""
    x, y, z = p
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


def torus(P, R, r):
    x, y, z = P
    q = np.sqrt(x * x + y * y) - R
    return np.sqrt(q * q + z * z) - r


def ellipsoid(P, d):
    x, y, z = P
    a, b, c = d[0] / 2, d[1] / 2, d[2] / 2
    k0 = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
    k1 = np.sqrt((x / a ** 2) ** 2 + (y / b ** 2) ** 2 + (z / c ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def node_field(n, X, Y, Z):
    if n.get("mx"): X = np.abs(X)
    if n.get("my"): Y = np.abs(Y)
    if n.get("mz"): Z = np.abs(Z)
    p = rot((X - n["p"][0], Y - n["p"][1], Z - n["p"][2]), n["r"])
    if n["t"] == "torus":
        return torus(p, n["d"][0], n["d"][1])
    if n["t"] == "ellipsoid":
        return ellipsoid(p, n["d"])
    if n["t"] == "plane":
        # SDF of the region the cut removes: everything below the plane
        return (Z - n["p"][2]).astype(F32)
    raise ValueError(n["t"])


def build(doc, body_id, X, Y, Z):
    """Bodies are built in node order: `add` nodes whose `b` is this body,
    `cut` nodes whose `tg` lists it (or is null, meaning every body)."""
    d = None
    for n in doc["nodes"]:
        if not n.get("on", True):
            continue
        op, b, tg = n["op"], n.get("b", 0), n.get("tg")
        if op == "add":
            if b != body_id:
                continue
        elif tg is not None and body_id not in tg:
            continue                      # tg == [] targets nothing
        f = node_field(n, X, Y, Z)
        k = float(n.get("k", 0))
        if op == "add":
            d = f if d is None else smin(d, f, k)
        elif d is not None:
            d = smax(d, -f, k)
    return d
