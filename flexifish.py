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

Two linkages are available (`joint_style`): the ball-and-socket one this
started with, assembled per joint out of a ball, a neck and a shell; and the
interlocking-ring one from joint_tool.py, which is a single solid subtracted
from the finished body. See that module for why the second one scales more
easily and what it costs -- chiefly length, since it needs a good deal more
room along the body than a ball does.
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

import joint_tool

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
    # joint_style picks the linkage:
    #   "ball"  a plate-cut ball captured in a spherical socket. Compact --
    #           a joint costs about 2*Rb of body length, so ~10 mm -- and it
    #           can be popped apart under force, which is what joint_capture
    #           tunes. Everything under "segment joints" below applies.
    #   "tool"  two interlocking rings, made by subtracting joint_tool.py's
    #           solid. Cannot be pulled apart at all, and needs nothing solved
    #           against the local section, but it is long: about 34 mm of body
    #           per joint at full size, so far fewer segments fit. Sizing
    #           reduces n_segments rather than produce a fish that comes out
    #           in one piece.
    joint_style: str = "ball"          # "ball" | "tool" | "none"
    joint_gap: float = 0.0             # "tool": fixed clearance at every
                                       # joint; 0 keeps the tool's own, which
                                       # tapers with the body (~1.2 -> 0.6 mm)
    # Where the ring cutter sits, on top of the fit it works out for itself.
    # The defaults are that automatic fit; these are for driving it by hand
    # when the automatic one lands somewhere you would rather it did not --
    # too far into a segment, too deep, too big for the section. Ignored by
    # the other linkages, which have nothing to place.
    tool_offset: float = 0.0           # slide it along the body, mm
    tool_lift: float = 0.0             # raise it off the build plate, mm
    tool_scale: float = 1.0            # x the automatic section fit
    n_segments: int = 5                # articulated segments between head and tail piece
    head_length: float = 34.0          # nose -> first joint
    tail_root_len: float = 12.0        # solid peduncle length ahead of the caudal fin
    swing_front_deg: float = 16.0      # per-joint swing at the front...
    swing_rear_deg: float = 12.0       # ...tapering to this at the tail
    clearance: float = 0.55            # radial joint clearance (PETG ~0.55, PLA ~0.45)
    face_gap: float = 1.0              # min gap between segment faces at the centerline
    wall: float = 1.6                  # socket wall thickness
    min_seg_len: float = 6.0           # refuse to make segments shorter than this

    # ---- segment joints: plate-cut ball captured in a spherical socket --
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
    # One pair, a separate print-in-place part each: a plate-cut ball captured
    # in a spherical socket, riding the dorsal fin's segment. Set the length
    # to 0 to remove the pair.
    #
    # There used to be a pectoral pair on the head as well. Two pairs is what
    # a fish has and one pair is what this prints well: the head pair sat far
    # enough forward that its sockets pushed the first joint cut back, which
    # cost a segment, and on the plate the two pairs' paddles competed for the
    # same skirt of free space beside the body.
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


# Parameters that used to be here. A config written before the pectoral fins
# came off still describes a fish; it just says something about a fin that is
# no longer there. Dropping those keys with a word beats refusing to build.
RETIRED_PARAMS = {"pec_length", "pec_width", "pec_thickness", "pec_pos",
                  "pec_sweep_deg"}


def load_overrides(path):
    """Read a --config file into a FishParams keyword dict."""
    over = json.load(open(path))
    gone = sorted(set(over) & RETIRED_PARAMS)
    for k in gone:
        over.pop(k)
    if gone:
        print(f"ignoring retired parameter(s) in {path}: {gone} "
              f"-- the pectoral fins were removed")
    unknown = sorted(set(over) - {f.name for f in fields(FishParams)})
    if unknown:
        sys.exit(f"unknown parameter(s) in {path}: {unknown}")
    return over


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
        if p.joint_style not in ("ball", "tool", "none"):
            raise SystemExit(f"unknown joint_style {p.joint_style!r} "
                             "(expected 'ball', 'tool' or 'none')")
        if p.joint_style == "tool":
            self.p = p = self._fit_tool_segments(p)
        self._layout()
        self._size_joints()
        # after sizing, not before: how much body a joint disturbs is not
        # known until it has been sized, and a ring joint reaches far
        self._check_side_fins()

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
        if p.joint_style == "none":
            # a fish in one piece: no cuts, so no joints, and the dorsal fin
            # is free to sit wherever it was drawn
            self.cuts = np.array([], dtype=float)
            self.dorsal_on_head = False
            self.dorsal_trim = (-1e4, 1e4)
            self.p = replace(p, n_segments=0)
            return
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

    def _joint_spans(self):
        """How much of the body's *skin* each joint disturbs, as (x0, x1).

        A ball joint is a hollow on the centreline: the only thing it does
        out at the surface is cut the body in two, so its span is the cut
        plane and the gap either side of it. The ring tool is a solid
        subtracted from the whole section, shroud and all -- it eats
        `FOOTPRINT_AHEAD` in front of the joint and `FOOTPRINT_BEHIND`
        behind, about 26 mm of body at full size, most of it on one side.
        Anything at the surface that has to survive intact keeps clear of
        these."""
        p = self.p
        out = []
        for j in self.joints:
            if p.joint_style == "tool":
                out.append((j["xt"] - j["ahead"], j["xt"] + j["behind"]))
            else:
                out.append((j["xa"] - p.face_gap / 2, j["xa"] + p.face_gap / 2))
        return out

    def _check_side_fins(self):
        """Every side-fin socket has to sit inside ONE piece of the body.

        The socket is a hollow, not a bump: a joint reaching into it leaves
        two half-sockets, one either side of a joint that then moves, and the
        fin falls out. The pelvic pair rides the dorsal fin's segment, which
        is the shortest piece on the fish, so this is worth saying plainly
        rather than leaving it to the shell count at the end."""
        spans = self._joint_spans()
        if not spans:
            return                                # one solid fish, no joints
        inf = float("inf")
        edges = [(-inf, -inf)] + spans + [(inf, inf)]
        for g in self._fins():
            x, r = g["cx"], g["boss_r"] + 1.0
            hit = [(a, b) for a, b in spans if a < x + r and b > x - r]
            if not hit:
                continue
            free = [(a[1], b[0]) for a, b in zip(edges, edges[1:])
                    if b[0] - a[1] > 2 * r]

            def show(w):
                a, b = w
                if a == -inf:
                    return f"ahead of x = {b - r:.1f}"
                if b == inf:
                    return f"behind x = {a + r:.1f}"
                return f"x = {a + r:.1f} .. {b - r:.1f}"

            free.sort(key=lambda w: min(abs(w[0] - x), abs(w[1] - x)))
            where = ("; it fits " + " or ".join(show(w) for w in free[:2])
                     if free else
                     "; no piece of this fish is long enough for it -- "
                     "lengthen the body, or draw a narrower paddle (the "
                     "socket is sized from the outline's chord)")
            if self.p.joint_style == "tool":
                where += (". A ring joint takes far more body than a ball "
                          "does, so joint_style='ball' may be the easier fix "
                          "than moving the fin")
            raise SystemExit(
                f"the {g.get('name', 'side fin')} socket at x = {x:.1f} "
                f"(radius {g['boss_r']:.1f} mm) runs into the joint that "
                f"takes x = {hit[0][0]:.1f} .. {hit[0][1]:.1f}, so it would "
                f"print in halves and the fin would fall out{where}.")

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

    def _wall_for(self, R):
        """Socket wall thins with the ball so small joints stay printable."""
        return min(self.p.wall, max(1.0, 0.45 * R))

    def _size_joints(self):
        if self.p.joint_style == "none":
            self.joints = []
            return
        if self.p.joint_style == "tool":
            return self._size_tool_joints()
        p = self.p
        cuts = self.cuts
        segl = np.diff(cuts)
        n = len(cuts)
        top = [self.top_at(x) for x in cuts]
        self.joints = []
        for i, xa in enumerate(cuts):
            # Largest ball whose shell fits inside the body at its own
            # height, under the back, and end-to-end with the neighbouring
            # joint's shell. Every term depends on R, so bisect.
            def fits(R):
                Rb = R + p.clearance + self._wall_for(R)
                zc = self._ball_zc(R, p.clearance)
                if Rb > self.halfwidth_at(xa, zc) - 0.7:
                    return False
                if zc + Rb > top[i] - 0.9:
                    return False
                for s in (i - 1, i):          # adjacent articulated segments
                    if 0 <= s < len(segl) and 2 * Rb > segl[s] - 2.4:
                        return False          # 2.4 = clearance + solid bridge
                return True
            lo, hi = 0.4, p.joint_ball_max
            if fits(hi):
                R = hi
            else:
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid
                R = lo
            we = self._wall_for(R)
            if R < 1.6:
                raise SystemExit(
                    f"joint at x={xa:.1f} would need a ball of only "
                    f"R={R:.2f} mm (<1.6 minimum). Segments are too short "
                    f"or the body too narrow there -- reduce n_segments, "
                    f"lengthen the fish, or widen the body.")
            if R < 2.2:
                self.warnings.append(
                    f"joint at x={xa:.1f}: delicate (ball R={R:.1f} mm) -- "
                    f"handle gently, or widen the body")

            zc = self._ball_zc(R, p.clearance)
            rn = max(1.0, p.joint_neck * R)
            Rb = R + p.clearance + we
            # Socket mouth: open it only as far as the requested swing needs.
            # Aperture (R+clearance)*sin(theta) must stay under the ball
            # radius; what is left over is the grip that resists pop-out.
            t = i / max(n - 1, 1)
            sw_req = (p.swing_front_deg
                      + (p.swing_rear_deg - p.swing_front_deg) * t)
            half_neck = np.rad2deg(np.arcsin(np.clip(rn / R, 0.02, 0.95)))
            grip_min = max(p.joint_capture * R, 0.45)   # never a token lip
            th_max = np.rad2deg(np.arcsin(np.clip(
                (R - grip_min) / (R + p.clearance), 0.05, 0.97)))
            th = min(sw_req + half_neck, th_max)
            sw = max(0.0, th - half_neck)
            grip = R - (R + p.clearance) * np.sin(np.deg2rad(th))
            if sw < sw_req - 0.5:
                self.warnings.append(
                    f"joint at x={xa:.1f}: swing limited to +/-{sw:.0f} deg "
                    f"(requested {sw_req:.0f}) to keep the ball captured; "
                    f"a wider body allows more swing")
            if grip < 0.45:
                self.warnings.append(
                    f"joint at x={xa:.1f}: only {grip:.2f} mm of socket grip "
                    f"-- may pop apart easily; lower swing_*_deg or raise "
                    f"joint_capture")
            beta = sw / 2.0 + 2.0
            self.joints.append(dict(
                xa=float(xa), top=top[i], R=float(R), zc=float(zc),
                rn=float(rn), Rb=float(Rb), swing=float(sw),
                theta=float(th), grip=float(grip),
                cosO=float(np.cos(np.deg2rad(th))),
                nl=float(Rb + 2.2),               # neck length, ball -> body
                above=float(self._above_plate_frac(R, zc)),
                tanb=float(np.tan(np.deg2rad(beta)))))

    # ---------------- joint sizing: subtracted interlocking rings -----
    def _tool_segment_cap(self, first, last, n):
        """The most segments a ring joint will allow between two stations.

        A ball joint is small enough that the layout never has to know it is
        there. This one is not: it eats about `joint_tool.MIN_SPACING` mm of
        body, so a fish that takes five ball joints takes one or two of these.
        Cutting the count down and saying so beats honouring it and handing
        back a fish whose segments came out fused to each other.

        Sized on the section at `first`, where the body is deepest and the
        tool therefore largest. Since the layouts space their cuts evenly, one
        conservative figure is the right shape of answer."""
        need = (joint_tool.MIN_SPACING * self.p.tool_scale
                * joint_tool.scales_for(self, first)[1])
        m = max(1, min(n, int((last - first) // need)))
        if m != n:
            self.warnings.append(
                f"joint_style='tool': {n} segments would space the joints "
                f"{(last - first) / n:.1f} mm apart and a ring joint needs "
                f"{need:.1f} mm; building {m} instead. Lengthen the fish for "
                f"more, or use joint_style='ball'.")
        return m

    def _fit_tool_segments(self, p):
        return replace(p, n_segments=self._tool_segment_cap(
            p.head_length, self.L - p.tail_root_len, p.n_segments))

    def _size_tool_joints(self):
        """Nothing to solve -- the tool overshoots the body on purpose, so the
        only decisions are how much to scale it and how far apart the joints
        ended up. Both are recorded here for `plate` to use."""
        p = self.p
        segl = np.diff(self.cuts)
        n = len(self.cuts)
        self.joints = []
        for i, xa in enumerate(self.cuts):
            s_wide, s_long, s_tall = (
                v * p.tool_scale for v in joint_tool.scales_for(self, xa))
            ahead, behind = joint_tool.footprint(s_long)
            s = min(s_wide, s_long, s_tall)
            auto = joint_tool.THICKNESS * s
            self.joints.append(dict(
                xa=float(xa), top=self.top_at(xa),
                # where the cut goes vs where the tool goes: the segmentation
                # is decided by `xa`, and the solid may be placed off it
                xt=float(xa + p.tool_offset), lift=float(p.tool_lift),
                s_wide=float(s_wide), s_long=float(s_long),
                s_tall=float(s_tall),
                gap=float(p.joint_gap if p.joint_gap > 0 else auto),
                ahead=float(ahead), behind=float(behind)))
            if p.joint_gap > 0 and p.joint_gap > joint_tool.THICKNESS * s + 0.6:
                self.warnings.append(
                    f"joint at x={xa:.1f}: joint_gap={p.joint_gap:.2f} mm is "
                    f"far wider than the tool's own {auto:.2f} mm here; the "
                    f"rings are eroded that much too and may end up slack")
            if i == 0 and p.tool_lift > 0.05:
                # the tool's own plate is its bottom: nothing below it is ever
                # removed, so a lifted cutter leaves a continuous bridge of
                # body under every joint. Useful for looking at where the cut
                # sits, never something to print.
                self.warnings.append(
                    f"tool_lift={p.tool_lift:.2f} mm holds the cutter off the "
                    f"build plate, so {p.tool_lift:.2f} mm of body is left "
                    f"under every joint and the segments print as one piece")
            if self.joints[-1]["gap"] < 0.3:
                self.warnings.append(
                    f"joint at x={xa:.1f}: only {self.joints[-1]['gap']:.2f} mm "
                    f"of clearance -- the section is small, so the tool is "
                    f"small. Set joint_gap to pin it, or shorten the fish.")
            # The joints are laid out evenly, but the dorsal fin is centred in
            # its own segment whose length is fixed by the fin -- so one pair
            # can end up much closer than the average. Check the pairs.
            if i:
                got, want = segl[i - 1], self.joints[i - 1]["behind"] + ahead
                floor = joint_tool.SHATTER_SPACING * s_long
                if got + 1e-6 < floor:
                    raise SystemExit(
                        f"joints at x={self.cuts[i-1]:.1f} and x={xa:.1f} are "
                        f"only {got:.1f} mm apart; below {floor:.1f} mm the "
                        f"two cuts destroy each other's linkage and the fish "
                        f"comes out in loose pieces. Lengthen the fish, cut "
                        f"n_segments, or use joint_style='ball'.")
                if got + 1e-6 < want:
                    self.warnings.append(
                        f"joints at x={self.cuts[i-1]:.1f} and x={xa:.1f} are "
                        f"{got:.1f} mm apart but together reach {want:.1f} mm; "
                        f"that segment is all shroud, with none of its own "
                        f"body showing. Raise dorsal_length/fin_margin if it "
                        f"is the dorsal segment, or cut n_segments.")
        if n and self.top_at(self.cuts[-1]) < 6.0:
            self.warnings.append(
                f"the last joint sits where the body is only "
                f"{self.top_at(self.cuts[-1]):.1f} mm deep; a ring joint that "
                f"small is fragile -- raise tail_root_len to move it forward")

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
        """The pelvic pair, on the dorsal fin's segment.

        `len_nose_to_dorsal` is the centre of that segment by construction --
        `_layout` builds the segment around the fin -- so anchoring the balls
        there keeps both sockets as far from the two cut faces as the segment
        allows, whatever the layout does with the rest of the body."""
        p = self.p
        if p.pelvic_length <= 0.5:
            return []
        xb, span = p.len_nose_to_dorsal, p.pelvic_length
        chord, th, sw = p.pelvic_width, p.pelvic_thickness, p.pelvic_sweep_deg
        rb = min(max(0.42 * chord, 3.0), 5.0)
        # ball center high enough that the plate-cut socket lip still
        # wraps below the ball's equator -> captured in every direction
        zc = max(0.62 * rb, float(np.sqrt(2 * rb * p.clearance)) + 0.45)
        cy = self.halfwidth_at(xb, zc) - 0.2 * rb
        a = np.deg2rad(sw)
        return [dict(name="pelvic", cx=xb, cy=cy, cz=zc, rb=rb,
                     dx=float(np.sin(a)), dy=float(np.cos(a)),
                     span=span, chord=chord, th=th,
                     rn=0.45 * rb, boss_r=rb + p.clearance + p.wall)]

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
        # 1.0, not the 1.5 this used to be: the fillet where the neck
        # leaves the ball grows material *outward*, into the very clearance
        # the joint needs to turn. At 1.5 the ball-to-socket gap measured
        # 0.43 mm against a nominal 0.55, which is under two voxels at the
        # 0.3 mm print resolution -- meshers bridged it and the fin came out
        # welded to the body. At 1.0 the gap is 0.52 and the tightest place
        # is no longer the ball.
        f = smin(smin(ball, neck, F32(1.0)), pad, F32(2.0))
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

    # ---------------- joint geometry: ball, socket, shell -------------
    def _joint(self, j, X, Y, Z, want, x_front_limit=None):
        """All three pieces are concentric on the ball centre, so nothing
        sticks up on a stalk to be levered out, and the socket shell is a
        sphere that can be clipped to the body silhouette."""
        p, xa, zc, R = self.p, j["xa"], j["zc"], j["R"]
        px = (X - xa).astype(F32)
        pz = (Z - zc).astype(F32)
        r = np.sqrt(px * px + Y * Y + pz * pz).astype(F32)
        if want == "pin":                          # ball + neck, front side
            ball = (r - R).astype(F32)
            x0 = xa - j["nl"]
            if x_front_limit is not None:          # stay clear of the front
                x0 = min(max(x0, x_front_limit), xa - R - 0.4)
            L = max(xa - x0, 0.1)
            t = np.clip(-px / L, 0.0, 1.0)         # capsule toward -x
            qx = px + L * t
            neck = (np.sqrt(qx * qx + Y * Y + pz * pz) - j["rn"]).astype(F32)
            # the ball dips below z=0 by design; the plate cut is what
            # gives it a flat, unsupported-printable bottom
            return np.maximum(np.minimum(ball, neck), (-Z).astype(F32))
        if want == "cav":                          # socket cavity + mouth
            sph = (r - (R + p.clearance)).astype(F32)
            mouth = np.maximum((r - (j["Rb"] + 2.0)).astype(F32),
                               (j["cosO"] * r + px).astype(F32))
            return np.minimum(sph, mouth)
        if want == "boss":                         # spherical socket shell
            return (r - j["Rb"]).astype(F32)
        if want == "relief":                       # clearance for the shell
            return (r - (j["Rb"] + p.clearance + 0.35)).astype(F32)
        raise ValueError(want)

    # ---------------- assemble the printable plate --------------------
    def plate(self, X, Y, Z):
        F = self.styled(X, Y, Z)
        fins = self._fins()
        for g in fins:
            F = np.maximum(F, -self.fin_cavity(g, X, Y, Z))
        out = (F if self.p.joint_style == "none"
               else self._split_tool(F, X, Y, Z) if self.p.joint_style == "tool"
               else self._split_ball(F, X, Y, Z))
        for g in fins:
            for sgn in (1.0, -1.0):
                out = np.minimum(out, self.fin_part(g, sgn, X, Y, Z))
        return out

    def _split_tool(self, F, X, Y, Z):
        """Subtract one tool per joint. The body is already finished when this
        runs -- there is no per-segment assembly at all, which is the whole
        point of the thing."""
        p = self.p
        out = F
        for j in self.joints:
            out = np.maximum(out, -joint_tool.tool_sdf(
                X, Y, Z, xa=j["xt"], s_wide=j["s_wide"], s_long=j["s_long"],
                s_tall=j["s_tall"], lift=j["lift"],
                gap=p.joint_gap if p.joint_gap > 0 else None))
        return out

    def _split_ball(self, F, X, Y, Z):
        p = self.p
        absY = np.abs(Y)
        J = self.joints
        nseg = len(J) + 1
        out = None
        for i in range(nseg):
            seg = F.copy()
            front_limit, boss_own = None, None
            if i > 0:                              # socket at the front face
                jf = J[i - 1]
                bound = jf["xa"] + p.face_gap / 2 + jf["tanb"] * absY
                seg = np.maximum(seg, (bound - X).astype(F32))
                # clipped to the body: the shell can never bulge out of the
                # fish even where the section is tight. Clip 0.15 mm inside
                # the surface so the two never land exactly coincident,
                # which marching cubes would mesh as a degenerate crease.
                boss_own = np.maximum(self._joint(jf, X, Y, Z, "boss"),
                                      (F + F32(0.15)))
                seg = np.minimum(seg, boss_own)
                seg = np.maximum(seg, -self._joint(jf, X, Y, Z, "cav"))
                front_limit = jf["xa"] + jf["Rb"] + 0.6
            if i < nseg - 1:                       # ball + neck at the rear
                jr = J[i]
                bound = jr["xa"] - p.face_gap / 2 - jr["tanb"] * absY
                seg = np.maximum(seg, (X - bound).astype(F32))
                pin = self._joint(jr, X, Y, Z, "pin", x_front_limit=front_limit)
                # hollow out room for the next segment's shell to swing
                cut = self._joint(jr, X, Y, Z, "relief")
                cut = np.maximum(cut, -pin)        # never carve the ball/neck
                if boss_own is not None:           # ...or this segment's shell
                    cut = np.maximum(cut, -boss_own)
                seg = np.maximum(seg, -cut)
                seg = np.minimum(seg, pin)
            out = seg if out is None else np.minimum(out, seg)
        return out

    # ---------------- grid bounds -------------------------------------
    def bounds(self):
        p = self.p
        x0 = -0.2 * self.top_at(0.05 * self.L) - 3.0
        x1 = self.L + p.tail_length + p.tail_fork + 4.0
        yw = (p.body_width / 2
              + max(p.eye_proud, p.pelvic_length + 7.0)
              + 3.0)
        z1 = max(self.top_at(p.len_nose_to_dorsal) + p.dorsal_height,
                 self.top_at(0.2 * self.L), p.tail_height) + 3.0
        return x0, x1, -yw, yw, -1.2, z1


# ======================================================================
# Meshing, export, verification
# ======================================================================
def coupon_window(b, i):
    """x-range for a one-joint test print: (x0, x1, isolated).

    It has to hold the whole linkage, or the window cuts the coupon in two and
    the shell count means nothing. A ring joint is long enough that on a
    closely-jointed fish such a window also reaches into a neighbouring
    joint's shroud and slices a loose crescent off it, which shows up as an
    extra shell -- printable, and no reflection on the joint under test.
    Containing the linkage matters more, so the window is never narrowed;
    `isolated` says whether to expect the extra piece."""
    j = b.joints[i]
    if b.p.joint_style != "tool":
        return j["xa"] - 7.0, j["xa"] + 7.5, True
    lo, hi = j["xa"] - j["ahead"] - 2.0, j["xa"] + j["behind"] + 2.0
    ok = (i == 0 or b.joints[i - 1]["xa"] + b.joints[i - 1]["behind"] <= lo)
    ok &= (i + 1 == len(b.joints)
           or b.joints[i + 1]["xa"] - b.joints[i + 1]["ahead"] >= hi)
    return lo, hi, ok


def mesh(builder, res, field="plate", sub=None):
    x0, x1, y0, y1, z0, z1 = builder.bounds()
    if sub:
        x0, x1 = max(x0, sub[0] - 1.5), min(x1, sub[1] + 1.5)
    xs = np.arange(x0, x1 + res, res, dtype=F32)
    ys = np.arange(y0, y1 + res, res, dtype=F32)
    zs = np.arange(z0, z1 + res, res, dtype=F32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = builder.plate(X, Y, Z) if field == "plate" else builder.styled(X, Y, Z)
    if sub:
        vol = np.maximum(vol, sd_box(X, Y, Z, sub[0], sub[1],
                                     y0 - 5, y1 + 5, -5, z1 + 5))
    # Mesh a hair *inside* the surface, not on it. The build-plate cut is
    # max(d, -Z), which is exactly 0.0 everywhere the body crosses z = 0 -- and
    # z0 and res routinely put a lattice plane right there, so the belly comes
    # out as a sheet of thousands of exact zeros. Marching cubes has no way to
    # orient a cell whose corners are all zero, and the result is a belly that
    # welds neighbouring segments together: at level 0.0 the default fish
    # meshes as 6 shells, at -0.001 as the 8 it should be. The offset costs
    # half a micron of size.
    verts, faces, _, _ = measure.marching_cubes(vol, level=-1e-3,
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


def mesh_stats(verts, faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    manifold = bool(np.all(counts == 2))
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
    shells = len({find(v) for v in faces.ravel()})
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
        p = replace(p, **load_overrides(args.config))
    if args.res:
        p = replace(p, res=args.res)
    res = 0.62 if args.preview and not args.res else p.res

    t0 = time.time()
    b = FishBuilder(p)
    for w in b.warnings:
        print("note:", w)
    p = b.p                      # 'tool' joints may have cut n_segments down
    jxs = ", ".join(f"{j['xa']:.0f}" for j in b.joints)
    print(f"fish: {b.L + p.tail_length:.0f} mm long, {p.n_segments} "
          f"segments, {len(b.joints)} {p.joint_style} joints at x = {jxs}")

    verts, faces = mesh(b, res)
    write_stl(args.out, verts, faces)
    man, shells = mesh_stats(verts, faces)
    # a segmented fish is head + n segments + tail root; an unsegmented one is
    # a single body. The side fins are free either way.
    body_pieces = 1 if p.joint_style == "none" else p.n_segments + 2
    expected = body_pieces + (2 if p.pelvic_length > 0.5 else 0)
    print(f"{args.out}: {len(faces)} tris, manifold={man}, "
          f"shells={shells} (expected {expected})  [{time.time()-t0:.0f}s]")
    if shells != expected and not args.preview:
        print("WARNING: shell count mismatch -- parts may be fused or "
              "orphaned. Inspect before printing.")

    if args.coupon:
        i = len(b.joints) // 2
        lo, hi, isolated = coupon_window(b, i)
        cv, cf = mesh(b, res, sub=(lo, hi))
        cp = args.out.replace(".stl", "") + "_joint_test.stl"
        write_stl(cp, cv, cf)
        cm, cs = mesh_stats(cv, cf)
        note = "" if isolated else "; neighbouring joints reach into this one"
        print(f"{cp}: {len(cf)} tris, manifold={cm}, shells={cs} "
              f"(expected 2{note})")

    if args.png:
        pv, pf = (verts, faces) if res >= 0.5 else mesh(b, 0.55)
        png = args.out.replace(".stl", "") + "_views.png"
        render_png(png, pv, pf)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
