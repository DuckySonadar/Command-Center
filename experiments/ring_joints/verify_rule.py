#!/usr/bin/env python3
"""Verify the ring-pair design rule against brute-force measurement.

Rule (derived): with the tilted ring's axis at angle `a` from vertical in
the XZ plane and the vertical ring's axis along Y, centres offset by `off`
along x,

    links when  off < 2*R*cos(a)
    best at     off = R*cos(a)
    max centreline separation = R*(1 - sin a)
    so          tube r <= (R*(1 - sin a) - clearance) / 2

Run:  python3 verify_rule.py
"""
import numpy as np

from proto import ring_pts, linked

NB = np.array([0.0, 1.0, 0.0])          # vertical ring: axis along Y
CLEARANCE = 0.55


def axisA(a_deg):
    """Tilted ring's axis of revolution: in the XZ plane, a_deg from +Z."""
    a = np.deg2rad(a_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])


def min_sep(CA, NA, R, CB, NBv, RB, n=1400):
    A = ring_pts(CA, NA, R, n)
    B = ring_pts(CB, NBv, RB, n)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).min()


if __name__ == "__main__":
    print("predicted vs measured centreline separation at off = R*cos(a)\n")
    print(f"{'a':>4} {'off':>8} {'measured':>10} {'predicted':>10} {'linked':>7}")
    R = 4.0
    for a in (0, 10, 20, 30, 45, 60):
        off = R * np.cos(np.deg2rad(a))
        CA = np.array([-off / 2, 0.0, 5.0])
        CB = np.array([off / 2, 0.0, 5.0])
        NA = axisA(a)
        sep = min_sep(CA, NA, R, CB, NB, R)
        pred = R * (1 - np.sin(np.deg2rad(a)))
        ok, _ = linked(CA, NA, R, CB, NB, R)
        print(f"{a:4.0f} {off:8.2f} {sep:10.3f} {pred:10.3f} {str(ok):>7}")

    print("\nlargest usable tube radius r = (R*(1 - sin a) - clearance)/2")
    print("(below ~0.7 mm the ring is too fragile to trust)\n")
    print(f"{'a':>4} " + "".join(f"{'R='+str(R):>9}" for R in (4, 5, 6)))
    for a in (0, 15, 30, 45, 60):
        row = "".join(
            f"{(R * (1 - np.sin(np.deg2rad(a))) - CLEARANCE) / 2:9.2f}"
            for R in (4.0, 5.0, 6.0))
        print(f"{a:4.0f} " + row)

    print("\n-> a = 30 (ring PLANE 60 deg from vertical) is the only reading"
          "\n   that leaves a usable tube. a = 60 gives ~0.06 mm: impossible.")
