#!/usr/bin/env python3
"""
flexifish.py -- parametric print-in-place flexi fish generator.

Builds an articulated cartoon fish as a signed distance field (smooth blends
give the rounded "toy" look), meshes it with marching cubes, and writes a
binary STL ready to slice. Prints belly-down on the build plate (z = 0), no
supports. All dimensions are millimeters.

Quick start:
    python flexifish.py                          # default fish -> fish_plate.stl
    python flexifish.py --preview --png          # fast coarse build + render
    python flexifish.py --config my_fish.json    # override any parameters
    python flexifish.py --coupon                 # also emit a joint-test print
    python flexifish.py --dump-config            # write defaults to fish_defaults.json
    python flexifish.py --list-params            # show every knob

Requires: numpy, scikit-image (matplotlib only for --png).

Coordinate system: x runs nose -> tail, y is left/right, z is up.
The build plate is the z = 0 plane; the body is sunk `belly_drop` mm below
it and cut, which is what makes the flat belly every segment rests on.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass, asdict, fields, replace

import numpy as np

try:
    from skimage import measure
except ImportError:  # pragma: no cover
    sys.exit("flexifish needs scikit-image (pip install numpy scikit-image)")

F32 = np.float32


# ======================================================================
# Parameters
# ======================================================================
@dataclass
class FishParams:
    # ---- overall body (mm) -------------------------------------------
    len_nose_to_dorsal: float = 55.0   # nose tip -> center of dorsal fin
    len_dorsal_to_tail: float = 58.0   # dorsal center -> caudal fin root
    body_width: float = 40.0           # max body width (eyes excluded)
    body_height: float = 38.0          # max body height above build plate (fins excluded)
    belly_drop: float = 3.5            # how far the body sinks below the build
                                       # plate before the z=0 cut (bigger = wider flat belly)

    # ---- articulation ------------------------------------------------
    n_segments: int = 2                # articulated segments between head and tail piece
    head_length: float = 34.0          # nose -> first joint
    tail_root_len: float = 12.0        # solid peduncle length ahead of the caudal fin
    swing_front_deg: float = 16.0      # per-joint swing at the front...
    swing_rear_deg: float = 12.0       # ...tapering to this at the tail
    clearance: float = 0.55            # radial joint clearance (PETG ~0.55, PLA ~0.45)
    face_gap: float = 1.0              # min gap between segment faces at the centerline
    wall: float = 1.6                  # socket wall thickness
    min_seg_len: float = 6.0           # refuse to make segments shorter than this

    # ---- segment joints: two interlocked rings (topologically linked) ---
    # A tilted torus on the front segment threads a vertical torus on the
    # rear one. Being linked rather than clipped, capture does not depend on
    # tolerance at all -- the joint only comes apart if a ring breaks, and
    # nothing else is needed to hold the two segments together.
    # Geometry rule (derived + verified): with the tilted ring's axis at
    # `ring_axis_deg` from vertical in the XZ plane, the largest possible
    # centreline separation is R*(1 - sin a) at an offset of R*cos(a), so
    #     tube r <= (R*(1 - sin a) - clearance) / 2
    ring_axis_deg: float = 30.0        # tilted ring axis from vertical (XZ)
    ring_tube_min: float = 0.70        # thinnest acceptable tube radius
    ring_max: float = 7.0              # largest ring centreline radius
    # There is no dome. An earlier version nested a ball on the front segment
    # into a matching cup on the rear, as a second capture on top of the
    # rings -- but the rings are a link, they hold on their own, and a ball
    # big enough to matter is a ball big enough to swallow the tilted ring.
    # The seam is a plain wedge and the linkage carries the joint.

    # ---- legacy ball-joint knobs (unused by the ring linkage) ----------
    # The socket mouth is a cone whose aperture is deliberately NARROWER
    # than the ball: `joint_capture` is that interference as a fraction of
    # the ball radius. It is what makes a joint pop apart only under real
    # force. The mouth is opened just wide enough for the requested swing
    # and no wider, so grip is usually well above this floor.
    joint_ball_max: float = 5.0        # largest ball radius (mm)
    joint_capture: float = 0.20        # min grip (fraction of ball radius)
    joint_neck: float = 0.40           # neck radius / ball radius

    # ---- eyes ----------------------------------------------------------
    eye_diameter: float = 12.5
    eye_pos: float = 0.30              # fraction along the head (0 = nose, 1 = first joint)
    eye_height: float = 0.62           # fraction of local body height
    eye_proud: float = 3.2             # how far the dome bulges past the body surface
    pupil_diameter: float = 4.6        # dimple pupil (0 disables)

    # ---- dorsal fin (a "deltoid" fin: always centered in one segment) --
    dorsal_length: float = 16.0        # base length along the spine
    dorsal_height: float = 12.0        # rise above the local back line
    dorsal_rake: float = 0.55          # 0 = upright, 1 = strongly swept back
    dorsal_thickness: float = 2.8      # at the root (thins toward the tip)
    fin_margin: float = 2.2            # keep-out between a fin base and segment faces

    # ---- side fins: flat on the plate, ball-jointed to the body --------
    # Each is a separate print-in-place part: a plate-cut ball captured in a
    # spherical socket. Pectorals ride the head, pelvics ride the dorsal
    # fin's segment. Set a length to 0 to remove that pair.
    pec_length: float = 15.0
    pec_width: float = 9.5             # chord (fore-aft size of the paddle)
    pec_thickness: float = 3.6
    pec_pos: float = 0.62              # fraction along the head
    pec_sweep_deg: float = 32.0        # swept back toward the tail
    pelvic_length: float = 13.0
    pelvic_width: float = 8.5
    pelvic_thickness: float = 3.4
    pelvic_sweep_deg: float = 30.0

    # ---- caudal (tail) fin ---------------------------------------------
    tail_length: float = 40.0          # peduncle end -> fin tip
    tail_height: float = 34.0          # vertical span of the fan
    tail_fork: float = 6.0             # fork notch depth (0 = rounded paddle)
    tail_thick_root: float = 4.0
    tail_thick_tip: float = 2.2

    # ---- fin cross-section ----------------------------------------------
    fin_edge: float = 0.9              # rim thickness of dorsal/tail fins (mm)
    fin_fillet: float = 2.8            # width of the taper from core to rim

    # ---- face ----------------------------------------------------------
    lip_size: float = 1.0              # pucker scale (0 disables lips)

    # ---- body sculpting / meshing --------------------------------------
    n_blobs: int = 4                   # body blobs; more = smoother profile
    head_size: float = 1.0             # head scale vs the body front (0.5-1.2)
    head_point: float = 0.35           # nose shape: 0 = spherical, 1 = pointed
    blend: float = 9.0                 # body blob blend radius (higher = doughier)
    res: float = 0.30                  # voxel size for meshing


# Body silhouette control points: (t along body, half-width frac,
# half-height frac). The chain resamples this at n_blobs stations.
PROFILE = [
    (0.30, 0.93, 0.95),
    (0.47, 0.84, 0.905),
    (0.71, 0.58, 0.667),
    (0.90, 0.395, 0.405),
]


def profile_at(t):
    if t <= PROFILE[0][0]:
        return PROFILE[0][1], PROFILE[0][2]
    for (t0, w0, h0), (t1, w1, h1) in zip(PROFILE, PROFILE[1:]):
        if t <= t1:
            a = (t - t0) / (t1 - t0)
            return w0 + (w1 - w0) * a, h0 + (h1 - h0) * a
    return PROFILE[-1][1], PROFILE[-1][2]


# ======================================================================
# SDF toolbox (all fields: negative inside, positive outside)
# ======================================================================
def smin(a, b, k):
    """Smooth union."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    """Smooth intersection (use smax(a, -b, k) for smooth subtraction)."""
    return -smin(-a, -b, k)


def sd_ellipsoid(X, Y, Z, cx, cy, cz, rx, ry, rz):
    px, py, pz = (X - cx) / rx, (Y - cy) / ry, (Z - cz) / rz
    k0 = np.sqrt(px * px + py * py + pz * pz)
    qx, qy, qz = px / rx, py / ry, pz / rz
    k1 = np.maximum(np.sqrt(qx * qx + qy * qy + qz * qz), 1e-9)
    return np.where(k0 > 1e-9, k0 * (k0 - 1.0) / k1,
                    -min(rx, ry, rz)).astype(F32)


def sd_circle2(u, v, cu, cv, r):
    return (np.sqrt((u - cu) ** 2 + (v - cv) ** 2) - r).astype(F32)


def sd_box(X, Y, Z, x0, x1, y0, y1, z0, z1):
    qx = np.maximum(x0 - X, X - x1)
    qy = np.maximum(y0 - Y, Y - y1)
    qz = np.maximum(z0 - Z, Z - z1)
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2
                  + np.maximum(qz, 0) ** 2)
    ins = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return (out + ins).astype(F32)


def sd_cyl_v(X, Y, cx, cy, r):
    """Infinite vertical cylinder."""
    return (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r).astype(F32)


def sd_torus(X, Y, Z, cx, cy, cz, nx, ny, nz, R, r):
    """Torus: centre (cx,cy,cz), axis of revolution (nx,ny,nz) unit, major
    radius R, tube radius r."""
    px, py, pz_ = X - cx, Y - cy, Z - cz
    h = px * nx + py * ny + pz_ * nz              # along the axis
    qx, qy, qz = px - h * nx, py - h * ny, pz_ - h * nz
    q = np.sqrt(qx * qx + qy * qy + qz * qz)      # radius from the axis
    return (np.sqrt((q - R) ** 2 + h * h) - r).astype(F32)


def sd_polygon(vr, vz, pr, pz):
    """2D polygon SDF (Inigo Quilez formulation), vectorized."""
    n = len(vr)
    d = (pr - vr[0]) ** 2 + (pz - vz[0]) ** 2
    s = np.ones_like(pr)
    j = n - 1
    for i in range(n):
        er, ez = vr[j] - vr[i], vz[j] - vz[i]
        wr, wz = pr - vr[i], pz - vz[i]
        t = np.clip((wr * er + wz * ez) / (er * er + ez * ez + 1e-12), 0.0, 1.0)
        br, bz = wr - er * t, wz - ez * t
        d = np.minimum(d, br * br + bz * bz)
        c1 = pz >= vz[i]
        c2 = pz < vz[j]
        c3 = er * wz > ez * wr
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
        j = i
    return (s * np.sqrt(d)).astype(F32)


def rot2(u, v, deg):
    a = np.deg2rad(deg)
    return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)


def fin_sheet(d2, Y, t_root, t_tip, u, u0, u1, edge=0.45, fillet=2.8):
    """Extrude a 2D silhouette SDF into a fin with a lens cross-section:
    full thickness in the core, tapering to `edge` half-thickness at the
    rim over a `fillet`-mm-wide blend (a filleted disk, not a slab)."""
    s = np.clip((u - u0) / max(u1 - u0, 1e-6), 0, 1)
    t = t_root + (t_tip - t_root) * s
    slope = (t_root - edge) / max(fillet, 0.3)
    th = np.minimum(t, edge + slope * np.maximum(-d2, 0.0))
    r = 0.6
    w = np.abs(Y) - (th - r)
    return (np.maximum(d2 + r, w) - r).astype(F32)


# ======================================================================
# The fish
# ======================================================================
class FishBuilder:
    def __init__(self, p: FishParams):
        self.p = p
        self.L = p.len_nose_to_dorsal + p.len_dorsal_to_tail  # nose -> caudal root
        self.warnings: list[str] = []
        self._layout()
        self._size_joints()

    # ---------------- core body (blobs + flat belly) ----------------
    def core(self, X, Y, Z):
        p = self.p
        ry0 = p.body_width / 2.0
        rz0 = (p.body_height + p.belly_drop) / 2.0
        n = max(2, int(round(p.n_blobs)))
        t0, t1 = 0.30, 0.90
        span = (t1 - t0) / (n - 1)
        rx = max(0.85 * span * self.L, 7.0)
        d = None
        for i in range(n):
            t = t0 + span * i
            fw, fh = profile_at(t)
            rz = rz0 * fh
            e = sd_ellipsoid(X, Y, Z, t * self.L, 0.0, rz - p.belly_drop,
                             rx, ry0 * fw, rz)
            d = e if d is None else smin(d, e, F32(p.blend))
        # parametric head: main ellipsoid + snout sphere; the blend between
        # them morphs the nose from spherical (0) to a drawn, pointed taper (1)
        hs, pt, hl = p.head_size, p.head_point, p.head_length
        ryh, rzh = ry0 * hs, rz0 * hs
        rxh = 0.62 * hl
        head = sd_ellipsoid(X, Y, Z, rxh + 1.0, 0.0, rzh - p.belly_drop,
                            rxh, ryh, rzh)
        rs = (0.80 - 0.52 * pt) * min(ryh, rzh)
        zn = 0.40 * (2 * rzh - p.belly_drop)
        snout = sd_ellipsoid(X, Y, Z, rs + 0.6, 0.0, zn, rs, rs, rs)
        head = smin(head, snout, F32((4.0 + 9.0 * pt) * min(hl / 34.0, 1.6)))
        return smin(d, head, F32(p.blend))

    # ---------------- numeric probes on the core --------------------
    def top_at(self, x):
        zs = np.linspace(0, 120, 1201, dtype=F32)
        f = self.core(np.full_like(zs, x), np.zeros_like(zs), zs)
        return float(zs[f < 0].max()) if np.any(f < 0) else 0.0

    def halfwidth_at(self, x, z):
        ys = np.linspace(0, 80, 1601, dtype=F32)
        f = self.core(np.full_like(ys, x), ys, np.full_like(ys, z))
        return float(ys[f < 0].max()) if np.any(f < 0) else 0.0

    # ---------------- segmentation layout ---------------------------
    def _layout(self):
        p = self.p
        first, last = p.head_length, self.L - p.tail_root_len
        xd = p.len_nose_to_dorsal
        wd = p.dorsal_length + 2 * p.fin_margin
        n = p.n_segments
        if n < 1:
            raise SystemExit("n_segments must be >= 1")
        if last - first < n * p.min_seg_len:
            raise SystemExit(
                f"body too short for {n} segments: articulated span is "
                f"{last - first:.1f} mm, max segments = "
                f"{int((last - first) // p.min_seg_len)}")

        self.dorsal_on_head = xd - wd / 2 <= first + 0.5
        if self.dorsal_on_head:
            cuts = list(np.linspace(first, last, n + 1))
            self.dorsal_trim = (2.0, first - p.face_gap / 2 - 1.0)
        else:
            a, b = xd - wd / 2, xd + wd / 2
            if b > last - 0.5:
                raise SystemExit("dorsal fin lands in the tail piece; reduce "
                                 "len_nose_to_dorsal or dorsal_length")
            rem = n - 1
            nb = int(round(rem * (a - first) / max((a - first) + (last - b), 1e-6)))
            nb = max(0, min(rem, nb))
            while nb > 0 and (a - first) / nb < p.min_seg_len:
                nb -= 1
            while (rem - nb) > 0 and (last - b) / (rem - nb) < p.min_seg_len:
                nb += 1
            nb = max(0, min(rem, nb))
            if nb == 0:                      # widen dorsal segment forward,
                a = first                    # keeping the fin at its center
                b = 2 * xd - a
            if rem - nb == 0:
                b = last
                a = 2 * xd - b
            if a < first - 1e-6 or b > last + 1e-6:
                raise SystemExit("dorsal fin too close to head/tail to center "
                                 "it in a segment; adjust lengths")
            cuts = (list(np.linspace(first, a, nb + 1))
                    + list(np.linspace(b, last, rem - nb + 1)))
            self.dorsal_trim = (a + p.face_gap / 2 + 1.0,
                                b - p.face_gap / 2 - 1.0)
        self.cuts = np.array(sorted(set(np.round(cuts, 4))))
        if len(self.cuts) != p.n_segments + 1:
            raise SystemExit("internal layout error: bad cut count")
        if np.any(np.diff(self.cuts) < p.min_seg_len - 1e-6):
            raise SystemExit(
                f"segments as short as {np.diff(self.cuts).min():.1f} mm; "
                f"reduce n_segments or move the dorsal fin")

    # ---------------- joint sizing: plate-cut ball and socket ---------
    @staticmethod
    def _ball_zc(R, clearance):
        """Ball centre height. Low enough that the ball rests on the plate
        and prints unsupported, high enough that the socket lip still wraps
        below its equator -- and always >80% of the ball above z = 0."""
        return max(0.55 * R, float(np.sqrt(2 * R * clearance)) + 0.45)

    @staticmethod
    def _above_plate_frac(R, zc):
        """Fraction of the ball's volume above the build plate."""
        h = max(min(R - zc, 2 * R), 0.0)          # spherical cap below z = 0
        return 1.0 - (h * h * (3 * R - h)) / (4 * R ** 3)

    # ---------------- joint sizing: interlocked ring pair -------------
    @staticmethod
    def _ring_tube(R, a_deg, clearance):
        """Largest tube radius that still leaves `clearance` between the two
        centrelines. From R*(1 - sin a) being the maximum separation."""
        sep = R * (1.0 - np.sin(np.deg2rad(a_deg)))
        return (sep - clearance) / 2.0

    def _ring_layout(self, R):
        """Everything that follows from the ring radius. One place, because
        `fits()` and the accepted joint must agree exactly -- when they did
        not, the search approved sizes the build then rejected."""
        p = self.p
        ca = np.cos(np.deg2rad(p.ring_axis_deg))
        r = self._ring_tube(R, p.ring_axis_deg, p.clearance)
        off = R * ca
        # Ring height above the plate. The *other* segment's ring is carved
        # out of this one with `clearance` of relief, so the floor left under
        # that carve is (zc - R - r - clearance); keep 0.6 mm of it or the
        # segment ends in a wafer that will not print.
        zc = R + r + p.clearance + 0.6
        return r, off, zc

    def _size_joints(self):
        p = self.p
        cuts = self.cuts
        segl = np.diff(cuts)
        n = len(cuts)
        top = [self.top_at(x) for x in cuts]
        a = p.ring_axis_deg
        ca, sa = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
        self.joints = []
        for i, xa in enumerate(cuts):
            # Largest ring pair that fits: the linkage spans off + 2(R+r)
            # along the body, both rings have to sit inside the section with
            # skin left over them, and they must clear the build plate.
            def fits(R):
                r, off, zc = self._ring_layout(R)
                if r < p.ring_tube_min:
                    return False
                # Ring envelopes. The vertical ring is the tall one (it
                # reaches zc +/- (R + r)); the tilted ring is the wide one
                # (+/- (R + r) in y, at z = zc, which is the body's widest
                # height). Both are clipped to the skin when the plate is
                # assembled, and a clipped ring is a broken ring, so they have
                # to fit with room to spare.
                if zc + R + r > top[i] - 0.8:
                    return False
                if R + r > self.halfwidth_at(xa, zc) - 0.8:
                    return False
                span = off + 2 * (R + r)
                for s_ in (i - 1, i):
                    if 0 <= s_ < len(segl) and span > segl[s_] - 2.0:
                        return False
                return True
            lo, hi = 0.8, p.ring_max
            if fits(hi):
                R = hi
            else:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid
                R = lo
            r, off, zc = self._ring_layout(R)
            if r < p.ring_tube_min:
                raise SystemExit(
                    f"joint at x={xa:.1f}: no ring pair fits (tube would be "
                    f"{max(r, 0):.2f} mm, minimum {p.ring_tube_min}). The "
                    f"linkage needs about {R * ca + 2 * (R + p.ring_tube_min):.0f} mm "
                    f"of segment length -- use fewer segments, a longer body, "
                    f"or lower ring_axis_deg.")
            t = i / max(n - 1, 1)
            sw = (p.swing_front_deg
                  + (p.swing_rear_deg - p.swing_front_deg) * t)
            if r < p.ring_tube_min + 0.15:
                self.warnings.append(
                    f"joint at x={xa:.1f}: thin rings (tube r={r:.2f} mm) -- "
                    f"they will not pull apart but can snap; longer segments "
                    f"allow thicker rings")
            self.joints.append(dict(
                xa=float(xa), top=top[i], R=float(R), rt=float(r),
                off=float(off), zc=float(zc),
                swing=float(sw),
                # Tilted ring: axis in XZ, `a` from vertical; sits forward.
                #
                # The sign of the x component decides which half of the ring
                # is the half that protrudes past the seam, and it is not a
                # free choice. Mirroring x swaps the two rings' positions, and
                # the vertical ring is symmetric about its own centre, so both
                # signs give an identical link and an identical separation --
                # but only one of them prints. With +sa the protruding arc is
                # the ring's *low* half, and its bottom (z = zc - R sin a - rt)
                # sits out past the seam face with nothing under it: the
                # printer would have to start it in mid-air. With -sa the
                # protruding arc is the high half, springing from the body at
                # z = zc + R sin a / 2 and arching up. Same ring, same joint,
                # supported instead of floating.
                axA=(float(-sa), 0.0, float(ca)),
                cA=(float(xa - off / 2), 0.0, float(zc)),
                # vertical ring: axis along Y; sits aft
                axB=(0.0, 1.0, 0.0),
                cB=(float(xa + off / 2), 0.0, float(zc)),
                tanb=float(np.tan(np.deg2rad(sw / 2 + 2.0)))))

    # ---------------- fins & face -----------------------------------
    def with_fins(self, X, Y, Z):
        p, L = self.p, self.L
        d = self.core(X, Y, Z)

        # caudal fan
        Ht, Lt = p.tail_height, p.tail_length
        zped = self.top_at(L - 2.0)
        c2 = smin(sd_circle2(X, Z, L - 0.10 * Lt, 0.65 * zped, 0.18 * Ht),
                  sd_circle2(X, Z, L + 0.34 * Lt, 0.44 * Ht, 0.20 * Ht),
                  F32(0.45 * Ht))                  # root + bridge
        c2 = smin(c2, sd_circle2(X, Z, L + 0.72 * Lt, 0.71 * Ht, 0.29 * Ht),
                  F32(0.45 * Ht))
        c2 = smin(c2, sd_circle2(X, Z, L + 0.66 * Lt, 0.03 * Ht, 0.28 * Ht),
                  F32(0.45 * Ht))
        if p.tail_fork > 0:
            rn = p.tail_fork + 3.0
            c2 = smax(c2, -sd_circle2(X, Z, L + Lt + (rn - p.tail_fork),
                                      0.37 * Ht, rn), F32(4.0))
        d = smin(d, fin_sheet(c2, Y, p.tail_thick_root, p.tail_thick_tip,
                              X, L - 0.1 * Lt, L + Lt,
                              p.fin_edge / 2, p.fin_fillet), F32(6.0))

        # dorsal fin -- centered in its segment (or on the head), and trimmed
        # so its base can never straddle a joint cut
        xd, Ld, Hd = p.len_nose_to_dorsal, p.dorsal_length, p.dorsal_height
        back = self.top_at(xd)
        # swept BACK: apex circle at the rear of the base, leading edge
        # lowered by rake so the fin rises toward the tail
        r_le, r_te = 0.40 * Ld, 0.33 * Ld
        z_top = back + Hd
        d2 = smin(sd_circle2(X, Z, xd - 0.5 * Ld + 0.9 * r_le,
                             z_top - r_le - p.dorsal_rake * 0.45 * Hd, r_le),
                  sd_circle2(X, Z, xd + 0.5 * Ld - 0.9 * r_te,
                             z_top - r_te, r_te),
                  F32(0.5 * Ld))
        lo, hi = self.dorsal_trim
        d2 = smax(d2, (lo - X).astype(F32), F32(1.5))
        d2 = smax(d2, (X - hi).astype(F32), F32(1.5))
        d = smin(d, fin_sheet(d2, Y, p.dorsal_thickness,
                              0.65 * p.dorsal_thickness, Z, back, z_top,
                              p.fin_edge / 2, p.fin_fillet), F32(5.0))

        # side-fin socket bosses (the fins are separate ball-jointed parts;
        # see _fins / fin_part / fin_cavity)
        for g in self._fins():
            for sgn in (1.0, -1.0):
                boss = (np.sqrt((X - g["cx"]) ** 2 + (Y - sgn * g["cy"]) ** 2
                                + (Z - g["cz"]) ** 2) - g["boss_r"])
                d = smin(d, boss.astype(F32), F32(3.0))
        return d

    # ---------------- side fins: flat, plate-cut ball joints ----------
    def _fins(self):
        p, out = self.p, []
        for which in ("pec", "pel"):
            if which == "pec":
                if p.pec_length <= 0.5:
                    continue
                xb, span = p.pec_pos * p.head_length, p.pec_length
                chord, th, sw = p.pec_width, p.pec_thickness, p.pec_sweep_deg
            else:
                if p.pelvic_length <= 0.5:
                    continue
                xb, span = p.len_nose_to_dorsal, p.pelvic_length
                chord, th = p.pelvic_width, p.pelvic_thickness
                sw = p.pelvic_sweep_deg
            rb = min(max(0.42 * chord, 3.0), 5.0)
            # ball center high enough that the plate-cut socket lip still
            # wraps below the ball's equator -> captured in every direction
            zc = max(0.62 * rb, float(np.sqrt(2 * rb * p.clearance)) + 0.45)
            cy = self.halfwidth_at(xb, zc) - 0.2 * rb
            a = np.deg2rad(sw)
            out.append(dict(cx=xb, cy=cy, cz=zc, rb=rb,
                            dx=float(np.sin(a)), dy=float(np.cos(a)),
                            span=span, chord=chord, th=th,
                            rn=0.45 * rb, boss_r=rb + p.clearance + p.wall))
        return out

    def fin_part(self, g, sgn, X, Y, Z):
        """One fin: plate-cut ball + neck + flat paddle, swept tailward."""
        cy, dy = sgn * g["cy"], sgn * g["dy"]
        ball = (np.sqrt((X - g["cx"]) ** 2 + (Y - cy) ** 2
                        + (Z - g["cz"]) ** 2) - g["rb"]).astype(F32)
        zf = g["th"] / 2 - 0.2
        nl = g["rb"] + 1.6
        ax, ay, az = g["dx"] * nl, dy * nl, zf - g["cz"]
        al2 = ax * ax + ay * ay + az * az
        t = np.clip(((X - g["cx"]) * ax + (Y - cy) * ay
                     + (Z - g["cz"]) * az) / al2, 0, 1)
        neck = (np.sqrt((X - g["cx"] - ax * t) ** 2 + (Y - cy - ay * t) ** 2
                        + (Z - g["cz"] - az * t) ** 2) - g["rn"]).astype(F32)
        fcx = g["cx"] + g["dx"] * (nl + g["span"] / 2 - 1.2)
        fcy = cy + dy * (nl + g["span"] / 2 - 1.2)
        a_ = (X - fcx) * dy - (Y - fcy) * g["dx"]
        b_ = (X - fcx) * g["dx"] + (Y - fcy) * dy
        pad = sd_ellipsoid(a_, b_, Z, 0.0, 0.0, zf,
                           g["chord"] / 2, g["span"] / 2, g["th"] / 2)
        f = smin(smin(ball, neck, F32(1.5)), pad, F32(2.0))
        return np.maximum(f, (-Z).astype(F32))

    def fin_cavity(self, g, X, Y, Z):
        """Socket cavity for both sides: dilated sphere + opening cone."""
        p, cav = self.p, None
        cosO = float(np.cos(np.deg2rad(48.0)))
        for sgn in (1.0, -1.0):
            cy, dy = sgn * g["cy"], sgn * g["dy"]
            px, py, pz = X - g["cx"], Y - cy, Z - g["cz"]
            r = np.sqrt(px * px + py * py + pz * pz)
            sph = (r - (g["rb"] + p.clearance)).astype(F32)
            opening = np.maximum(
                (r - (g["boss_r"] + 2.0)).astype(F32),
                (cosO * r - (px * g["dx"] + py * dy)).astype(F32))
            c = np.minimum(sph, opening)
            cav = c if cav is None else np.minimum(cav, c)
        return cav

    def preview(self, X, Y, Z):
        """Styled body with fins welded on (no clearances); for tests."""
        d = self.styled(X, Y, Z)
        for g in self._fins():
            for sgn in (1.0, -1.0):
                d = np.minimum(d, self.fin_part(g, sgn, X, Y, Z))
        return d

    def styled(self, X, Y, Z):
        p = self.p
        d = self.with_fins(X, Y, Z)

        # eyes: placed relative to the *measured* body surface
        xe = p.eye_pos * p.head_length
        ze = p.eye_height * self.top_at(xe)
        re = p.eye_diameter / 2
        hw = self.halfwidth_at(xe, ze)
        ye = hw + p.eye_proud - re
        for sgn in (1.0, -1.0):
            d = smin(d, sd_ellipsoid(X, Y, Z, xe, sgn * ye, ze, re, re, re),
                     F32(3.0))
        if p.pupil_diameter > 0:
            rp = p.pupil_diameter / 2
            look = np.array([-0.28, 0.92, 0.20])
            look /= np.linalg.norm(look)
            for sgn in (1.0, -1.0):
                c = (np.array([xe, sgn * ye, ze])
                     + look * np.array([1, sgn, 1]) * (re - 0.42 * rp))
                d = smax(d, -sd_ellipsoid(X, Y, Z, *c, rp, rp, rp), F32(0.9))

        # lips
        if p.lip_size > 0:
            tf = self.top_at(0.05 * self.L)
            R = 0.155 * tf * p.lip_size
            q = np.sqrt(Y ** 2 + ((Z - 0.34 * tf) / 0.85) ** 2) - R
            lips = np.sqrt(q * q + (X - 0.35 * R) ** 2) - 0.5 * R
            d = smin(d, lips.astype(F32), F32(3.0))

        # flat belly: the build-plate cut
        return np.maximum(d, (-Z).astype(F32))

    # ---------------- joint geometry: rings + seam face ---------------
    def _joint(self, j, X, Y, Z, want):
        """ringA = tilted torus (front segment), ringB = vertical torus (rear
        segment); `faceF` / `faceR` are the two sides of the seam, a wedge
        that opens with |y| so the segments can yaw past each other."""
        if want == "ringA":
            cx, cy, cz = j["cA"]; nx, ny, nz = j["axA"]
            return sd_torus(X, Y, Z, cx, cy, cz, nx, ny, nz, j["R"], j["rt"])
        if want == "ringB":
            cx, cy, cz = j["cB"]; nx, ny, nz = j["axB"]
            return sd_torus(X, Y, Z, cx, cy, cz, nx, ny, nz, j["R"], j["rt"])
        if want in ("faceF", "faceR"):
            # The seam: a plane through the joint that opens into a wedge with
            # |y|. The front segment keeps everything below -face_gap/2 of its
            # field, the rear everything above +face_gap/2 of its own, so the
            # two stay face_gap apart measured perpendicular to the surface --
            # hence the division by the gradient's length, which turns the
            # wedge from an offset along x into a distance.
            #
            # The wedge is what lets the segment yaw. The two sides need
            # opposite tilts -- both faces sloping the same way is a parallel
            # gap, and the rear sweeps straight into the front on the side it
            # turns toward. Sharing one field between them did exactly that
            # and held the joint to about 6 of its 14 degrees. The apex is the
            # vertical line through the joint centre, which is the axis the
            # joint turns about, so the V stays symmetric through the swing.
            dx = (X - j["xa"]).astype(F32)
            tb = F32(j["tanb"] if want == "faceF" else -j["tanb"])
            return ((dx + tb * np.abs(Y)) / np.sqrt(1.0 + tb * tb)).astype(F32)
        raise ValueError(want)

    def _swept(self, j, X, Y, Z, want):
        """The other segment's ring as it sweeps, not as it rests.

        The joint's motion is a yaw about the vertical line through the seam
        centre, and the ring pair barely notices it: the gap between the two
        centrelines holds at 0.55 mm from 0 to 20 degrees. The relief carved
        for the neighbouring ring, though, was a constant `clearance` offset
        around where that ring *sits*. The ring could therefore move
        `clearance` and then hit the wall of its own channel, which capped the
        joint near 4 degrees of the 14 it was sized for. Carving the swept
        volume instead costs a few extra torus evaluations and gives back the
        full range.

        A linked pair also slides, so the real motion is a yaw plus some
        translation. The sweep does not model that; the slack the rings have
        inside the relief (`clearance` of it, ~4.5 degrees' worth at R = 7) is
        what covers it, and the measured range in check_swing.py is the number
        that matters.
        """
        half = np.deg2rad(j["swing"] / 2.0 + 1.0)
        # 4-degree steps; the scallop left between samples is under 0.01 mm
        # anywhere near the rings, against 0.55 mm of clearance
        n = max(3, int(np.ceil(np.rad2deg(2 * half) / 4.0)) | 1)
        out = None
        for th in np.linspace(half, -half, n):     # inverse-rotate the samples
            c, s = np.cos(th), np.sin(th)
            dx = X - j["xa"]
            Xr = (j["xa"] + c * dx - s * Y).astype(F32)
            Yr = (s * dx + c * Y).astype(F32)
            v = self._joint(j, Xr, Yr, Z, want)
            out = v if out is None else np.minimum(out, v)
        return out

    # ---------------- assemble the printable plate --------------------
    def body_field(self, X, Y, Z, side_fins=True):
        """The whole body with the fin sockets hollowed out, before it is cut
        into segments. Every segment is carved from this."""
        F = self.styled(X, Y, Z)
        for g in (self._fins() if side_fins else []):
            F = np.maximum(F, -self.fin_cavity(g, X, Y, Z))
        return F

    def segment(self, i, X, Y, Z, F=None, side_fins=True):
        """Segment `i` on its own, from the head (0) back. Kept separate from
        `plate` so that the swing and clearance checks measure the same field
        the mesh is built from -- a second copy of this in a test harness is
        exactly how the swept relief came to look like it did nothing."""
        p = self.p
        F = self.body_field(X, Y, Z, side_fins) if F is None else F
        J = self.joints
        nseg = len(J) + 1
        seg = F.copy()
        if i > 0:                              # rear side of joint jf:
            jf = J[i - 1]                      # vertical ring side
            fv = self._joint(jf, X, Y, Z, "faceR")
            seg = np.maximum(seg, (F32(p.face_gap / 2) - fv))
            # clear room for the other segment's tilted ring to swing
            seg = np.maximum(seg, -(self._swept(jf, X, Y, Z, "ringA")
                                    - F32(p.clearance)))
            ring = np.maximum(self._joint(jf, X, Y, Z, "ringB"),
                              F + F32(0.15))   # never break the skin
            seg = np.minimum(seg, ring)
        if i < nseg - 1:                       # front side of joint jr:
            jr = J[i]                          # tilted ring side
            fv = self._joint(jr, X, Y, Z, "faceF")
            seg = np.maximum(seg, fv + F32(p.face_gap / 2))
            seg = np.maximum(seg, -(self._swept(jr, X, Y, Z, "ringB")
                                    - F32(p.clearance)))
            ring = np.maximum(self._joint(jr, X, Y, Z, "ringA"),
                              F + F32(0.15))
            seg = np.minimum(seg, ring)
        return seg

    def plate(self, X, Y, Z, side_fins=True):
        fins = self._fins() if side_fins else []
        F = self.body_field(X, Y, Z, side_fins)
        out = None
        for i in range(len(self.joints) + 1):
            seg = self.segment(i, X, Y, Z, F=F)
            out = seg if out is None else np.minimum(out, seg)
        for g in fins:
            for sgn in (1.0, -1.0):
                out = np.minimum(out, self.fin_part(g, sgn, X, Y, Z))
        return out

    # ---------------- grid bounds -------------------------------------
    def bounds(self):
        p = self.p
        x0 = -0.2 * self.top_at(0.05 * self.L) - 3.0
        x1 = self.L + p.tail_length + p.tail_fork + 4.0
        yw = (p.body_width / 2
              + max(p.eye_proud, max(p.pelvic_length, p.pec_length) + 7.0)
              + 3.0)
        z1 = max(self.top_at(p.len_nose_to_dorsal) + p.dorsal_height,
                 self.top_at(0.2 * self.L), p.tail_height) + 3.0
        return x0, x1, -yw, yw, -1.2, z1


# ======================================================================
# Meshing, export, verification
# ======================================================================
def mesh(builder, res, field="plate", box=None, side_fins=True):
    """`box` = (x0, x1, y0, y1, z0, z1) trims the build to a test coupon; the
    belly is never cut, so the piece still sits flat on the plate."""
    x0, x1, y0, y1, z0, z1 = builder.bounds()
    if box:
        x0, x1 = max(x0, box[0] - 1.5), min(x1, box[1] + 1.5)
        y0, y1 = max(y0, box[2] - 1.5), min(y1, box[3] + 1.5)
        z1 = min(z1, box[5] + 1.5)
    xs = np.arange(x0, x1 + res, res, dtype=F32)
    ys = np.arange(y0, y1 + res, res, dtype=F32)
    zs = np.arange(z0, z1 + res, res, dtype=F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = (builder.plate(X, Y, Z, side_fins=side_fins) if field == "plate"
           else builder.styled(X, Y, Z))
    if box:
        vol = np.maximum(vol, sd_box(X, Y, Z, box[0], box[1], box[2], box[3],
                                     -5.0, box[5]))
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0,
                                                spacing=(res, res, res))
    verts += np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
    return verts, faces


def write_stl(path, verts, faces):
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                      ("attr", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(b"flexifish parametric model".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        f.write(rec.tobytes())


def _components(verts, faces):
    """Union-find over shared vertices; returns a root label per face."""
    parent = np.arange(len(verts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        a = find(tri[0])
        for k in (1, 2):
            b = find(tri[k])
            if a != b:
                parent[b] = a
    return np.array([find(t) for t in faces[:, 0]])


def drop_debris(verts, faces, res):
    """Discard shells smaller than four voxels across.

    Where the ring relief channel leaves the body at a glancing angle the two
    surfaces cross in a feather edge thinner than a voxel, and marching cubes
    turns each stray sample into a lone octahedron. They are smaller than a
    nozzle, print as nothing, and their only real effect is to make the shell
    count -- the check that catches genuinely fused parts -- unreadable."""
    roots = _components(verts, faces)
    keep = np.ones(len(faces), dtype=bool)
    dropped = 0
    for r in np.unique(roots):
        m = roots == r
        v = verts[faces[m].ravel()]
        if (v.max(0) - v.min(0)).max() < 4 * res:
            keep[m] = False
            dropped += 1
    if not dropped:
        return verts, faces, 0
    faces = faces[keep]
    used = np.unique(faces)
    remap = np.zeros(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], dropped


def mesh_stats(verts, faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    manifold = bool(np.all(counts == 2))
    shells = len(np.unique(_components(verts, faces)))
    return manifold, shells


def render_png(path, verts, faces, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    lo, hi = verts.min(0), verts.max(0)
    c, span = (lo + hi) / 2, (hi - lo).max() * 0.62
    fig = plt.figure(figsize=(14, 9), dpi=125)
    for k, (e, a, t) in enumerate([(16, -152, "3/4 front"), (90, -90, "top"),
                                   (2, -90, "side"), (24, 38, "3/4 rear")]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        tri = verts[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        light = np.array([0.45, -0.6, 0.66])
        light /= np.linalg.norm(light)
        lam = np.clip(nrm @ light, 0, 1) * 0.75 + 0.25
        cols = np.clip(lam[:, None] * np.array([0.95, 0.55, 0.25]), 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols,
                                             edgecolor="none"))
        for setl, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setl(c[ci] - span, c[ci] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=e, azim=a)
        ax.set_axis_off()
        ax.set_title(f"{title} {t}".strip(), fontsize=12)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ======================================================================
# CLI
# ======================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file overriding any FishParams")
    ap.add_argument("--out", default="fish_plate.stl", help="output STL path")
    ap.add_argument("--res", type=float, help="voxel size override (mm)")
    ap.add_argument("--preview", action="store_true",
                    help="fast coarse build (joints may fuse at this res; "
                         "use full res for printable output)")
    ap.add_argument("--coupon", action="store_true",
                    help="also write joint_test.stl (print this first!)")
    ap.add_argument("--png", action="store_true",
                    help="also render a 4-view PNG next to the STL")
    ap.add_argument("--dump-config", action="store_true",
                    help="write all defaults to fish_defaults.json and exit")
    ap.add_argument("--list-params", action="store_true")
    args = ap.parse_args(argv)

    if args.dump_config:
        json.dump(asdict(FishParams()), open("fish_defaults.json", "w"),
                  indent=2)
        print("wrote fish_defaults.json")
        return
    if args.list_params:
        for f in fields(FishParams):
            print(f"{f.name:22s} = {f.default}")
        return

    p = FishParams()
    if args.config:
        overrides = json.load(open(args.config))
        unknown = set(overrides) - {f.name for f in fields(FishParams)}
        if unknown:
            sys.exit(f"unknown parameter(s) in {args.config}: {sorted(unknown)}")
        p = replace(p, **overrides)
    if args.res:
        p = replace(p, res=args.res)
    res = 0.62 if args.preview and not args.res else p.res

    t0 = time.time()
    b = FishBuilder(p)
    for w in b.warnings:
        print("note:", w)
    jxs = ", ".join(f"{j['xa']:.0f}" for j in b.joints)
    print(f"fish: {b.L + p.tail_length:.0f} mm long, {p.n_segments} "
          f"segments, {len(b.joints)} joints at x = {jxs}")

    verts, faces = mesh(b, res)
    verts, faces, junk = drop_debris(verts, faces, res)
    if junk:
        print(f"dropped {junk} sub-voxel speck(s)")
    write_stl(args.out, verts, faces)
    man, shells = mesh_stats(verts, faces)
    expected = (p.n_segments + 2 + (2 if p.pec_length > 0.5 else 0)
                + (2 if p.pelvic_length > 0.5 else 0))
    print(f"{args.out}: {len(faces)} tris, manifold={man}, "
          f"shells={shells} (expected {expected})  [{time.time()-t0:.0f}s]")
    if shells != expected and not args.preview:
        print("WARNING: shell count mismatch -- parts may be fused or "
              "orphaned. Inspect before printing.")

    if args.coupon:
        # A block around one linkage and nothing else: wide enough to hold
        # the whole ring pair (they reach off/2 + R + rt either side of the
        # joint), trimmed in y and z and built without side fins so the only
        # shells in it are the two segments.
        mid = b.joints[len(b.joints) // 2]
        reach = mid["off"] / 2 + mid["R"] + mid["rt"]
        cv, cf = mesh(b, res, side_fins=False, box=(
            mid["xa"] - reach - 2.0, mid["xa"] + reach + 2.0,
            -(mid["R"] + mid["rt"] + 2.5), mid["R"] + mid["rt"] + 2.5,
            0.0, mid["zc"] + mid["R"] + mid["rt"] + 2.5))
        cv, cf, _ = drop_debris(cv, cf, res)
        cp = args.out.replace(".stl", "") + "_joint_test.stl"
        write_stl(cp, cv, cf)
        cm, cs = mesh_stats(cv, cf)
        print(f"{cp}: {len(cf)} tris, manifold={cm}, shells={cs} (expected 2)")

    if args.png:
        pv, pf = (verts, faces) if res >= 0.5 else mesh(b, 0.55)
        png = args.out.replace(".stl", "") + "_views.png"
        render_png(png, pv, pf)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
