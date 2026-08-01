"""Corrected: tilted ring's AXIS lies in XZ; vertical ring's axis along Y."""
import numpy as np
from proto import ring_pts, linked, torus_sdf

CLR = 0.55
NB = np.array([0.0, 1.0, 0.0])            # vertical ring: axis along Y

def axisA(a_deg):                          # axis in the XZ plane, a from +Z
    a = np.deg2rad(a_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])

def min_dist(CA, NA, RA, CB, NB_, RB, n=900):
    A = ring_pts(CA, NA, RA, n); B = ring_pts(CB, NB_, RB, n)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).min()

def rotz(P, C, deg):
    r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r); v = P - C
    return C + np.stack([v[...,0]*c - v[...,1]*s, v[...,0]*s + v[...,1]*c, v[...,2]], -1)

def max_overhang(NA, R, r, n=600):
    """Steepest downward surface on the tilted torus, and how much area is
    within 5 deg of it (the 'thin rim')."""
    # surface normal at tube angle p around the centreline point
    C = ring_pts(np.zeros(3), NA, R, n)
    tang = np.gradient(C, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    radial = C / np.linalg.norm(C, axis=1, keepdims=True)
    binorm = np.cross(tang, radial)
    ph = np.linspace(0, 2*np.pi, 180, endpoint=False)
    nrm = (np.cos(ph)[None,:,None]*radial[:,None,:] +
           np.sin(ph)[None,:,None]*binorm[:,None,:])
    nz = nrm[..., 2].ravel()
    down = nz < 0
    # overhang measured from vertical wall: 0 = wall, 90 = flat ceiling
    ang = np.degrees(np.arcsin(np.clip(-nz[down], 0, 1)))
    return ang.max(), 100.0 * (ang > ang.max() - 5).mean()

print(f"{'axis':>5} {'plane':>6} {'linked':>7} {'R':>5} {'tube':>5} {'off':>5} "
      f"{'gap':>5} {'swing':>6} {'maxOH':>6} {'rim%':>5}")
for a in (15, 25, 30, 35, 45, 60):
    NA = axisA(a)
    hit = None
    for R in np.arange(2.2, 4.61, 0.2):
        r = 0.34 * R
        for off in np.arange(0.5*R, 1.6*R, 0.1):
            CA = np.array([-off/2, 0., 5.0]); CB = np.array([off/2, 0., 5.0])
            ok, _ = linked(CA, NA, R, CB, NB, R)
            if not ok: continue
            gap = min_dist(CA, NA, R, CB, NB, R) - 2*r
            if gap < CLR: continue
            piv = np.array([0., 0., 5.0]); lim = 60.0
            for sw in np.arange(0, 61, 2.0):
                CBr = rotz(CB, piv, sw); NBr = rotz(NB + piv, piv, sw) - piv
                if min_dist(CA, NA, R, CBr, NBr, R, 400) - 2*r < CLR*0.5:
                    lim = sw; break
            hit = (R, r, off, gap, lim); break
        if hit: break
    oh, rim = max_overhang(NA, 3.0, 1.0)
    if hit:
        R, r, off, gap, lim = hit
        print(f"{a:5.0f} {90-a:6.0f} {'yes':>7} {R:5.2f} {r:5.2f} {off:5.2f} "
              f"{gap:5.2f} {lim:5.0f}° {oh:5.1f}° {rim:5.1f}")
    else:
        print(f"{a:5.0f} {90-a:6.0f} {'no fit':>7} {'':>5} {'':>5} {'':>5} "
              f"{'':>5} {'':>6} {oh:5.1f}° {rim:5.1f}")
