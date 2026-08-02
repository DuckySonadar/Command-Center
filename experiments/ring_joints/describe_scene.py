"""Report the structure of an editor scene: what each body is made of, in what
order, how far the segments interleave, and how the cavity relates to the rings.

    python3 describe_scene.py [scene.json]

Written for porting the owner's reference joint into the generator. The things
it measures are the ones that turned out to matter and that reading the file
does not make obvious -- above all the *order* nodes are applied in, because a
shape added after a cut is immune to it, and that is what lets the seam cavity
be body-scale without destroying the linkage it passes through.
"""
import json
import os
import sys

import numpy as np

import sdf_json

HERE = os.path.dirname(os.path.abspath(__file__))


def ell_sdf(P, c, a):
    q = np.asarray(P) - c
    k0 = np.linalg.norm(q / a, axis=-1)
    k1 = np.linalg.norm(q / (a * a), axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def ring_pts(c, n, R, m=1500):
    a = np.array([0.0, 0.0, 1.0])
    if abs(n @ a) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    t = np.linspace(0, 2 * np.pi, m)
    return c + R * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)


def axis_of(deg):
    n = np.array([0.0, 0.0, 1.0])
    for ax, a in zip((0, 1, 2), np.deg2rad(deg)):
        c, s = np.cos(a), np.sin(a)
        if ax == 0:
            n = np.array([n[0], c * n[1] - s * n[2], s * n[1] + c * n[2]])
        elif ax == 1:
            n = np.array([c * n[0] + s * n[2], n[1], -s * n[0] + c * n[2]])
        else:
            n = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]])
    return n


def main(argv):
    scene = argv[1] if len(argv) > 1 else os.path.join(HERE, "owner_joint.json")
    doc = json.load(open(scene))
    nodes = doc["nodes"]
    ids = [b["id"] for b in doc["bodies"] if b.get("on", True)
           and any(n["op"] == "add" and n.get("b") == b["id"] for n in nodes)]
    print(f"{os.path.basename(scene)}: bodies {ids}\n")

    for bid in ids:
        print(f"body {bid} is built in this order:")
        for i, n in enumerate(nodes):
            if not n.get("on", True):
                continue
            if n["op"] == "add" and n.get("b") == bid:
                tag = "ADD"
            elif n["op"] == "cut" and (n.get("tg") is None
                                       or bid in (n.get("tg") or [])):
                tag = "CUT"
            else:
                continue
            where = "global" if n.get("tg") is None and n["op"] == "cut" else \
                    f"y={n['p'][1]:g}"
            print(f"  {i:2d} {tag} {n['t']:9s} k={n['k']:<5} {where:>8}"
                  f"  d={n['d'][:2]}")
        print("  (anything added after a cut is immune to it)")
        print()

    # how far along the body each segment reaches, and how much they interleave
    gx = np.arange(-24, 24, 0.5); gz = np.arange(6, 48, 0.5)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    cell = 0.5 * 0.5
    ys = np.arange(-40, 70, 0.5)
    area = {}
    for bid in ids:
        area[bid] = np.array([(sdf_json.build(doc, bid, GX, np.full_like(GX, y),
                                              GZ) < 0).sum() * cell for y in ys])
    for bid in ids:
        m = area[bid] > 0.5
        print(f"body {bid} spans y {ys[m].min():7.1f} .. {ys[m].max():7.1f}   "
              f"peak section {area[bid].max():.0f} mm2")
    if len(ids) >= 2:
        a, b = ids[0], ids[1]
        ov = (area[a] > 0.5) & (area[b] > 0.5)
        if ov.any():
            print(f"\nthey interleave over y {ys[ov].min():.1f} .. "
                  f"{ys[ov].max():.1f}  = {ys[ov].max()-ys[ov].min():.1f} mm.")
            print("This is a socket, not a butt joint: neither segment ends in "
                  "a face at\na station. The rear one's nose lives inside the "
                  "front one's dish.")

    # the cavity, against the rings it passes through
    for bid in ids:
        own = [(i, n) for i, n in enumerate(nodes) if n.get("on", True)]
        cav = [(i, n) for i, n in own if n["t"] == "ellipsoid" and n["op"] == "cut"
               and bid in (n.get("tg") or [])]
        rings = [(i, n) for i, n in own if n["t"] == "torus" and n["op"] == "add"
                 and n.get("b") == bid]
        if not cav or not rings:
            continue
        ci, cn = cav[0]
        cc = np.array(cn["p"], float); ac = np.array(cn["d"], float) / 2
        print(f"\nbody {bid}: seam cavity (node {ci}) against its own rings")
        for ri, rn in rings:
            P = ring_pts(np.array(rn["p"], float), axis_of(rn["r"]), rn["d"][0])
            d = ell_sdf(P, cc, ac).min()
            order = "added after the cut, so untouched" if ri > ci else \
                    "added BEFORE the cut, so the cut applies"
            verdict = "inside the cavity" if d < rn["d"][1] else \
                      f"clear by {d - rn['d'][1]:.2f} mm"
            print(f"  node {ri} torus at y={rn['p'][1]:g}: centreline {d:+6.2f} mm "
                  f"-> {verdict}\n      {order}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
