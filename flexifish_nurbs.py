#!/usr/bin/env python3
"""
flexifish_nurbs.py -- NURBS shape control for the flexi fish generator.

Instead of sculpting the body from blobs, you DRAW the fish with NURBS
curves in two views and the generator derives everything else from them:

  side view (x = nose->tail, z = up, z=0 is the build plate)
      back        open curve, nose tip -> caudal root, the top silhouette
      belly       open curve, nose tip -> caudal root, the bottom line
                  (dip below z=0; the plate cut makes the flat belly)
      dorsal_fin  closed outline, drawn overlapping the back so it fuses
      caudal_fin  closed outline of the tail fan (draw the fork right in)

  top view (x = nose->tail, y = half-width, right side only -- mirrored)
      plan          open curve, nose tip -> caudal root, the half-width
      pectoral_fin  closed outline of the front paddle, drawn in place
      pelvic_fin    closed outline of the rear paddle, drawn in place

Named regions are derived from what you drew, not typed in:

      head    | dorsal |    tail    | caudal
   0 ---------+--------+------------+--------> x
      rigid     1 seg,   N segments   tail-root
      body      fin       (the only    + fan
      (+ eyes)  centered   variable    piece
                in it      count)

  * everything ahead of the dorsal fin outline is the HEAD region; it
    stays rigid and carries the eyes and the mouth
  * the lower head region holds the PECTORAL and PELVIC fin regions --
    both outlines must attach there (validated), because their ball
    sockets must not straddle a joint cut
  * the dorsal fin outline claims the DORSAL region: exactly one
    articulated segment with the fin centered in it
  * the TAIL region is the only one with a variable segment count
    (regions.tail_segments)

The slider concept stays: per-region sliders scale the drawn curves
(smoothly blended at region borders so the body never steps):

  sliders.head    .length .width .height .mouth_open
  sliders.dorsal  .length .width .height .fin_height
  sliders.tail    .length .width .height
  sliders.caudal  .length .height .thickness
  sliders.pectoral.length .width          (about the attachment point)
  sliders.pelvic  .length .width

Quick start:
    python flexifish_nurbs.py                        # default fish
    python flexifish_nurbs.py --preview --png --svg  # coarse + renders
    python flexifish_nurbs.py --dump-shape           # editable template
    python flexifish_nurbs.py --shape my.json --set tail.length=1.3 \
        --set dorsal.fin_height=1.4 --set regions.tail_segments=4

The shape JSON is deep-merged over the built-in default, so specify only
what you change; set a fin curve to null to delete that fin. Joint,
eye, wall etc. parameters still come from flexifish's FishParams
(--config / --list-params). All the print-in-place joint machinery is
inherited unchanged from flexifish.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, fields, replace, asdict

import numpy as np

from flexifish import (F32, FishBuilder, FishParams, coupon_window, fin_sheet,
                       mesh, mesh_stats, render_png, sd_ellipsoid, sd_polygon,
                       smax, smin, write_stl)


# ======================================================================
# NURBS evaluation (clamped open / periodic closed, rational)
# ======================================================================
def _de_boor(t, ctrl, knots, p):
    """Vectorized de Boor: evaluate a B-spline at all params in t."""
    k = np.searchsorted(knots, t, side="right") - 1
    k = np.clip(k, p, len(ctrl) - 1)
    d = np.stack([ctrl[k - p + j] for j in range(p + 1)])
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = k - p + j
            den = knots[i + p - r + 1] - knots[i]
            a = np.where(den > 1e-12, (t - knots[i]) / np.where(den > 1e-12, den, 1.0), 0.0)
            d[j] = (1.0 - a)[:, None] * d[j - 1] + a[:, None] * d[j]
    return d[p]


def nurbs_eval(ctrl, degree=3, weights=None, closed=False, n=200):
    """Sample a rational B-spline through `ctrl` ((m,2) array) at n params."""
    ctrl = np.asarray(ctrl, dtype=float)
    m = len(ctrl)
    if m < 2:
        raise SystemExit("a curve needs at least 2 control points")
    p = max(1, min(int(degree), m - 1))
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != m:
        raise SystemExit("weights must match the number of control points")
    ch = np.column_stack([ctrl * w[:, None], w])
    if closed:
        ch = np.vstack([ch, ch[:p]])
        knots = (np.arange(len(ch) + p + 1) - p) / m
        t = np.linspace(0.0, 1.0, n, endpoint=False)
    else:
        knots = np.concatenate([np.zeros(p),
                                np.linspace(0.0, 1.0, m - p + 1),
                                np.ones(p)])
        t = np.linspace(0.0, 1.0, n)
    out = _de_boor(t, ch, knots, p)
    return out[:, :2] / np.maximum(out[:, 2:3], 1e-12)


def sample_curve(spec, n):
    return nurbs_eval(spec["points"], spec.get("degree", 3),
                      spec.get("weights"), spec.get("closed", False), n)


# ======================================================================
# Default shape: a hand-drawn classic that matches the blob fish's size
# ======================================================================
DEFAULT_SHAPE = {
    "curves": {
        "back": {"points": [[0, 9], [1, 22], [12, 34], [27, 41], [46, 38],
                            [63, 31], [85, 19.5], [103, 13.8], [113, 13.0]]},
        "belly": {"points": [[0, 9], [1, 3], [8, -2.5], [25, -4], [55, -4],
                             [85, -3], [104, -1.5], [113, -1.0]]},
        "plan": {"points": [[0, 0.8], [2, 6], [12, 15], [27, 20.5], [46, 18.5],
                            [63, 13.5], [85, 8.2], [103, 6.3], [113, 5.9]]},
        "dorsal_fin": {"closed": True,
                       "points": [[46, 31], [48, 40], [53, 46], [59, 47],
                                  [64, 42], [65, 35], [63, 29], [54, 28]]},
        "caudal_fin": {"closed": True,
                       "points": [[105, 12], [112, 20], [124, 30], [140, 36],
                                  [150, 33], [148, 26], [137, 19.5], [137, 18.5],
                                  [148, 11], [149, 2], [138, -2], [120, -2],
                                  [107, 0]]},
        "pectoral_fin": {"closed": True,
                         "points": [[16, 13], [21, 16], [28, 22], [32, 26],
                                    [29, 28], [22, 26], [16, 20], [13, 15]]},
        "pelvic_fin": {"closed": True,
                       "points": [[34, 12], [39, 14], [45, 18], [49, 22],
                                  [46, 24], [40, 23], [34, 17], [31, 13]]},
        "mouth": {"points": [[1.0, 14.0], [0.6, 11.0], [2.0, 8.5],
                             [5.5, 7.0]]},
    },
    "regions": {"tail_segments": 3, "blend_mm": 8.0},
    "mouth": {"shape": "curve", "height": 0.0, "tilt": 0.0, "length": 14.0},
    "sliders": {
        "head": {"length": 1.0, "width": 1.0, "height": 1.0, "mouth_open": 0.0},
        "dorsal": {"length": 1.0, "width": 1.0, "height": 1.0, "fin_height": 1.0},
        "tail": {"length": 1.0, "width": 1.0, "height": 1.0},
        "caudal": {"length": 1.0, "height": 1.0, "thickness": 1.0},
        "pectoral": {"length": 1.0, "width": 1.0},
        "pelvic": {"length": 1.0, "width": 1.0},
    },
}


def deep_merge(base, over):
    """Merge `over` into `base`; a None value deletes the key (drop a fin)."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ======================================================================
# Shape resolution: sample curves, apply sliders, derive regions
# ======================================================================
@dataclass
class ResolvedShape:
    xs: np.ndarray                     # increasing x stations of the body
    ztop: np.ndarray                   # back line z(x)
    zbot: np.ndarray                   # belly line z(x)
    wid: np.ndarray                    # half-width w(x)
    b0: float                          # nose
    b1: float                          # head | dorsal boundary
    b2: float                          # dorsal | tail boundary
    b3: float                          # caudal root
    tail_segments: int
    dorsal: np.ndarray | None = None   # closed outline samples (x, z)
    caudal: np.ndarray | None = None   # closed outline samples (x, z)
    dorsal_root_z: float = 0.0
    dorsal_top_z: float = 0.0
    caudal_thick: float = 1.0          # thickness multiplier
    mouth_open: float = 0.0
    mouth: dict = field(default_factory=dict)      # shape/height/tilt/length
    mouth_curve: np.ndarray | None = None          # side-view samples (x, z)
    side_fins: list = field(default_factory=list)  # dicts, see below
    svg_curves: dict = field(default_factory=dict)


def _monotone(pts):
    """Sort curve samples by x and force strictly increasing stations."""
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    x = np.maximum.accumulate(pts[:, 0] + np.arange(len(pts)) * 1e-9)
    return x, pts[:, 1]


def _blend_field(x, b1, b2, v_head, v_dorsal, v_tail, blend):
    def step(edge):
        t = np.clip((x - (edge - blend / 2)) / max(blend, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    return v_head + (v_dorsal - v_head) * step(b1) + (v_tail - v_dorsal) * step(b2)


def _logistic01(t, k=9.0):
    """Logistic growth curve rescaled so f(0) = 0 and f(1) = 1."""
    s = 1.0 / (1.0 + np.exp(-k * (np.clip(t, 0.0, 1.0) - 0.5)))
    s0 = 1.0 / (1.0 + np.exp(k * 0.5))
    s1 = 1.0 / (1.0 + np.exp(-k * 0.5))
    return (s - s0) / (s1 - s0)


def _polyline_dist(px, pz, pts):
    """Unsigned distance from grid points to an open 2D polyline."""
    d2 = None
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        ex, ez = bx - ax, bz - az
        t = np.clip(((px - ax) * ex + (pz - az) * ez)
                    / max(ex * ex + ez * ez, 1e-12), 0.0, 1.0)
        q = (px - ax - ex * t) ** 2 + (pz - az - ez * t) ** 2
        d2 = q if d2 is None else np.minimum(d2, q)
    return np.sqrt(d2)


def _validate_spec(spec):
    bad = set(spec) - {"curves", "regions", "sliders", "mouth"}
    if bad:
        raise SystemExit(f"unknown top-level shape key(s): {sorted(bad)}")
    bad = set(spec.get("mouth", {})) - {"shape", "height", "tilt", "length"}
    if bad:
        raise SystemExit(f"unknown mouth key(s): {sorted(bad)}")
    shape = spec.get("mouth", {}).get("shape")
    if shape is not None and shape not in ("pucker", "groove", "curve"):
        raise SystemExit("mouth.shape must be 'pucker', 'groove' or 'curve'")
    bad = set(spec.get("curves", {})) - set(DEFAULT_SHAPE["curves"])
    if bad:
        raise SystemExit(f"unknown curve(s): {sorted(bad)} "
                         f"(valid: {sorted(DEFAULT_SHAPE['curves'])})")
    bad = set(spec.get("regions", {})) - {"tail_segments", "blend_mm",
                                          "head_end", "tail_start"}
    if bad:
        raise SystemExit(f"unknown regions key(s): {sorted(bad)}")
    for reg, names in spec.get("sliders", {}).items():
        if reg not in DEFAULT_SHAPE["sliders"]:
            raise SystemExit(f"unknown slider region '{reg}' "
                             f"(valid: {sorted(DEFAULT_SHAPE['sliders'])})")
        if not isinstance(names, dict):
            raise SystemExit(f"sliders.{reg} must be a table, e.g. "
                             f"{reg}.length=1.2")
        bad = set(names) - set(DEFAULT_SHAPE["sliders"][reg])
        if bad:
            raise SystemExit(
                f"unknown slider(s) {sorted(bad)} for region '{reg}' "
                f"(valid: {sorted(DEFAULT_SHAPE['sliders'][reg])})")


def resolve_shape(spec: dict, p: FishParams) -> ResolvedShape:
    _validate_spec(spec or {})
    spec = deep_merge(DEFAULT_SHAPE, spec or {})
    curves = spec.get("curves", {})
    regions = spec.get("regions", {})
    sl = spec.get("sliders", {})

    def slider(region, name, default=1.0):
        return float(sl.get(region, {}).get(name, default))

    for req in ("back", "plan"):
        if req not in curves:
            raise SystemExit(f"shape needs a '{req}' curve")

    back = sample_curve(curves["back"], 300)
    plan = sample_curve(curves["plan"], 300)
    if "belly" in curves:
        belly = sample_curve(curves["belly"], 300)
    else:
        belly = np.column_stack([back[:, 0],
                                 np.full(len(back), -p.belly_drop)])
    dorsal = (sample_curve(curves["dorsal_fin"], 64)
              if "dorsal_fin" in curves else None)
    caudal = (sample_curve(curves["caudal_fin"], 80)
              if "caudal_fin" in curves else None)

    b0 = float(min(back[:, 0].min(), belly[:, 0].min(), plan[:, 0].min()))
    b3 = float(min(back[:, 0].max(), belly[:, 0].max(), plan[:, 0].max()))

    # ---- region boundaries: drawn dorsal fin claims its span ----------
    if dorsal is not None:
        b1 = float(dorsal[:, 0].min()) - p.fin_margin
        b2 = float(dorsal[:, 0].max()) + p.fin_margin
    else:
        b1 = b0 + 0.40 * (b3 - b0)
        b2 = b1
    b1 = float(regions.get("head_end", b1))
    b2 = float(regions.get("tail_start", b2))
    b1 = min(max(b1, b0 + 5.0), b3 - 5.0)
    b2 = min(max(b2, b1), b3 - 5.0)

    # ---- length sliders: piecewise-linear x remap of the regions ------
    old_k = np.array([b0, b1, b2, b3])
    new_k = np.concatenate([[b0], b0 + np.cumsum(
        np.diff(old_k) * [slider("head", "length"),
                          slider("dorsal", "length"),
                          slider("tail", "length")])])

    def remap_x(x):
        x = np.asarray(x, dtype=float)
        out = np.interp(x, old_k, new_k)
        return np.where(x > b3, x + (new_k[3] - b3), out)

    # side fins: find where each outline attaches BEFORE any remap, so
    # the paddle translates rigidly instead of stretching with the head
    side_specs = []
    for name, thick in (("pectoral_fin", p.pec_thickness),
                        ("pelvic_fin", p.pelvic_thickness)):
        if name not in curves:
            continue
        pts = sample_curve(curves[name], 56)
        pen = np.interp(pts[:, 0], *_monotone(plan)) - pts[:, 1]
        att = pts[np.argmax(pen)].copy()
        if pen.max() < 0.3:
            raise SystemExit(f"{name} outline never reaches the body -- "
                             f"draw it overlapping the plan curve")
        side_specs.append((name, pts, att, float(thick)))

    for arr in (back, belly, plan):
        arr[:, 0] = remap_x(arr[:, 0])
    if dorsal is not None:
        dorsal[:, 0] = remap_x(dorsal[:, 0])
    if caudal is not None:
        caudal[:, 0] = remap_x(caudal[:, 0])
    b1n, b2n, b3n = float(new_k[1]), float(new_k[2]), float(new_k[3])

    # ---- width / height sliders: smooth per-region scale fields -------
    blend = float(regions.get("blend_mm", 8.0))
    xs_b, ztop = _monotone(back)
    xs_l, zbot = _monotone(belly)
    xs_p, wid = _monotone(plan)
    xs = np.linspace(b0, b3n, 400)
    ztop = np.interp(xs, xs_b, ztop)
    zbot = np.interp(xs, xs_l, zbot)
    wid = np.interp(xs, xs_p, wid)
    ztop_raw = ztop.copy()
    hf = _blend_field(xs, b1n, b2n, slider("head", "height"),
                      slider("dorsal", "height"), slider("tail", "height"),
                      blend)
    wf = _blend_field(xs, b1n, b2n, slider("head", "width"),
                      slider("dorsal", "width"), slider("tail", "width"),
                      blend)
    ztop, zbot, wid = ztop * hf, zbot * hf, wid * wf
    # the tips may converge to a point; the interior must stay open
    inner = (xs > b0 + 4.0) & (xs < b3n - 2.0)
    if np.any(ztop[inner] <= zbot[inner] + 0.5):
        i = int(np.argmin(np.where(inner, ztop - zbot, np.inf)))
        raise SystemExit(f"back dips below belly near x={xs[i]:.0f} -- "
                         f"check the side-view curves")
    zbot = np.minimum(zbot, ztop - 0.2)

    S = ResolvedShape(xs=xs, ztop=ztop, zbot=zbot, wid=np.maximum(wid, 0.3),
                      b0=b0, b1=b1n, b2=b2n, b3=b3n,
                      tail_segments=int(regions.get("tail_segments", 3)),
                      mouth_open=slider("head", "mouth_open", 0.0))

    # ---- dorsal fin outline: track the (possibly rescaled) back -------
    if dorsal is not None:
        xc = 0.5 * (dorsal[:, 0].min() + dorsal[:, 0].max())
        shift = (np.interp(xc, xs, ztop) - np.interp(xc, xs, ztop_raw))
        dorsal[:, 1] += shift
        zmin = dorsal[:, 1].min()
        dorsal[:, 1] = zmin + (dorsal[:, 1] - zmin) * slider("dorsal", "fin_height")
        S.dorsal = dorsal
        S.dorsal_root_z = float(np.interp(xc, xs, ztop))
        S.dorsal_top_z = float(dorsal[:, 1].max())

    # ---- caudal fin outline -------------------------------------------
    # length/height sliders act through a logistic falloff: no effect
    # where the fin meets the tail root (the fused zone stays put), full
    # effect at the farthest edge
    if caudal is not None:
        x0c, x1c = caudal[:, 0].min(), caudal[:, 0].max()
        w = _logistic01((caudal[:, 0] - x0c) / max(x1c - x0c, 1e-6))
        caudal[:, 0] += w * (slider("caudal", "length") - 1.0) * (caudal[:, 0] - b3n)
        caudal[:, 1] *= 1.0 + w * (slider("caudal", "height") - 1.0)
        S.caudal = caudal
        S.caudal_thick = slider("caudal", "thickness")

    # ---- mouth ---------------------------------------------------------
    mspec = spec.get("mouth", {})
    S.mouth = {"shape": str(mspec.get("shape", "curve")),
               "height": float(mspec.get("height", 0.0)),
               "tilt": float(mspec.get("tilt", 0.0)),
               "length": float(mspec.get("length", 14.0))}
    if S.mouth["shape"] == "curve" and "mouth" in curves:
        mc = sample_curve(curves["mouth"], 60)
        mc[:, 0] = remap_x(mc[:, 0])
        mc[:, 1] += S.mouth["height"]
        mid = mc[len(mc) // 2].copy()          # rotate about the arc midpoint
        a = np.deg2rad(S.mouth["tilt"])
        ca, sa = np.cos(a), np.sin(a)
        rx = mc[:, 0] - mid[0]
        rz = mc[:, 1] - mid[1]
        mc[:, 0] = mid[0] + rx * ca - rz * sa
        mc[:, 1] = mid[1] + rx * sa + rz * ca
        S.mouth_curve = mc
    elif S.mouth["shape"] == "curve":
        S.mouth["shape"] = "groove"            # no curve drawn: plain plane

    # ---- pectoral / pelvic paddles ------------------------------------
    for name, pts, att, thick in side_specs:
        region = name.split("_")[0]
        dx_att = float(remap_x(att[0]) - att[0])
        pts = pts + [dx_att, 0.0]
        att = att + [dx_att, 0.0]
        cen = pts.mean(axis=0)
        d = cen - att
        d /= max(np.hypot(*d), 1e-9)
        perp = np.array([-d[1], d[0]])
        rel = pts - att
        pts = (att + np.outer(rel @ d, d) * slider(region, "length")
               + np.outer(rel @ perp, perp) * slider(region, "width"))
        u = (pts - att) @ d
        v = (pts - att) @ perp
        S.side_fins.append(dict(
            name=region, poly=pts, att=att, dirv=d, thick=thick,
            span=float(u.max()), chord=float(v.max() - v.min())))

    # stash for the SVG template
    S.svg_curves = {
        "back": ("side", np.column_stack([xs, ztop])),
        "belly": ("side", np.column_stack([xs, zbot])),
        "plan": ("top", np.column_stack([xs, wid])),
    }
    if S.dorsal is not None:
        S.svg_curves["dorsal_fin"] = ("side", S.dorsal)
    if S.caudal is not None:
        S.svg_curves["caudal_fin"] = ("side", S.caudal)
    if S.mouth_curve is not None:
        S.svg_curves["mouth"] = ("side", S.mouth_curve)
    for g in S.side_fins:
        S.svg_curves[g["name"] + "_fin"] = ("top", g["poly"])
    return S


# ======================================================================
# The NURBS-shaped fish: swap the body/fins, inherit all the joints
# ======================================================================
class NurbsFishBuilder(FishBuilder):
    def __init__(self, p: FishParams, spec: dict):
        S = resolve_shape(spec, p)
        self.shape = S
        cx_d = 0.5 * (S.b1 + S.b2)
        tail_x1 = S.caudal[:, 0].max() if S.caudal is not None else S.b3
        p = replace(
            p,
            # pucker lips only in pucker mode; groove modes carve instead
            lip_size=p.lip_size if S.mouth.get("shape") == "pucker" else 0.0,
            head_length=S.b1 - S.b0,
            len_nose_to_dorsal=cx_d,
            len_dorsal_to_tail=S.b3 - cx_d,
            body_width=2.0 * float(S.wid.max()),
            body_height=float(S.ztop.max()),
            tail_length=float(tail_x1 - S.b3),
            tail_height=float(S.caudal[:, 1].max()) if S.caudal is not None
            else p.tail_height,
        )
        super().__init__(p)
        if (S.mouth.get("shape") in ("groove", "curve")
                and abs(S.mouth.get("tilt", 0.0)) > 45.0):
            self.warnings.append(
                f"mouth plane tilted {S.mouth['tilt']:.0f} deg from vertical "
                f"(>45): the groove may overhang and cause print errors")

    # ---------------- lofted body from the drawn silhouettes ----------
    def core(self, X, Y, Z):
        S = self.shape
        zt = np.interp(X, S.xs, S.ztop)
        zb = np.interp(X, S.xs, S.zbot)
        w = np.interp(X, S.xs, S.wid)
        b = np.maximum((zt - zb) / 2.0, 0.3)
        zc = (zt + zb) / 2.0
        py = Y / w
        pz = (Z - zc) / b
        k0 = np.sqrt(py * py + pz * pz)
        k1 = np.maximum(np.sqrt((py / w) ** 2 + (pz / b) ** 2), 1e-9)
        d = np.where(k0 > 1e-9, k0 * (k0 - 1.0) / k1,
                     -np.minimum(w, b)).astype(F32)
        d = smax(d, (S.b0 - X).astype(F32), F32(1.0))     # cap the nose...
        d = smax(d, (X - S.b3).astype(F32), F32(2.0))     # ...and caudal root
        return d

    def _fit_tool_segments(self, p):
        """No-op here: n_segments is not an input to this fish -- the regions
        and `regions.tail_segments` decide the cuts, so `_layout` does the
        capping instead."""
        return p

    # ---------------- region-driven segmentation ----------------------
    def _layout(self):
        p, S = self.p, self.shape
        last = S.b3 - p.tail_root_len
        nt = S.tail_segments
        if nt < 1:
            raise SystemExit("regions.tail_segments must be >= 1")
        if last - S.b2 < nt * p.min_seg_len:
            raise SystemExit(
                f"tail region too short for {nt} segments: it is "
                f"{last - S.b2:.1f} mm, max segments = "
                f"{max(int((last - S.b2) // p.min_seg_len), 0)}")

        # the head must keep the side-fin ball sockets clear of the first
        # joint; if the dorsal fin is drawn that far forward, the dorsal
        # region fuses into the rigid head (same rule as the blob fish)
        need = S.b0 + 12.0
        for g in self.shape.side_fins:
            rb = min(max(0.42 * g["chord"], 3.0), 5.0)
            need = max(need, g["att"][0] + rb + p.clearance + p.wall + 1.0)
        self.dorsal_on_head = (S.b1 < need
                               or S.b2 - S.b1 < p.min_seg_len)
        # the ring joint is long; the tail region may not hold as many of them
        # as it holds ball joints (the dorsal region's own pair is checked
        # afterwards, per pair, in _size_tool_joints)
        if p.joint_style == "tool":
            nt = self._tool_segment_cap(S.b2, last, nt)
        if self.dorsal_on_head:
            if S.b2 < need:
                raise SystemExit(
                    "pectoral/pelvic fins attach behind the first joint "
                    f"cut (x={S.b2:.1f}); draw them further forward in "
                    "the head region")
            cuts = list(np.linspace(S.b2, last, nt + 1))
            self.dorsal_trim = (2.0, S.b2 - p.face_gap / 2 - 1.0)
        else:
            cuts = [S.b1] + list(np.linspace(S.b2, last, nt + 1))
            self.dorsal_trim = (S.b1 + p.face_gap / 2 + 1.0,
                                S.b2 - p.face_gap / 2 - 1.0)
        self.p = replace(p, n_segments=len(cuts) - 1)
        self.cuts = np.array(sorted(set(np.round(cuts, 4))))
        if np.any(np.diff(self.cuts) < p.min_seg_len - 1e-6):
            raise SystemExit(
                f"segments as short as {np.diff(self.cuts).min():.1f} mm; "
                f"reduce regions.tail_segments or lengthen the tail region")

    def region_table(self):
        S, rows = self.shape, []
        rows.append(("head", S.b0, S.b1 if not self.dorsal_on_head else S.b2,
                     "rigid" + ("" if not self.dorsal_on_head
                                else ", dorsal fin on head")))
        for g in S.side_fins:
            x0, x1 = g["poly"][:, 0].min(), g["poly"][:, 0].max()
            rows.append((f"  {g['name']}", x0, x1,
                         f"ball socket at x={g['att'][0]:.0f}"))
        if not self.dorsal_on_head:
            rows.append(("dorsal", S.b1, S.b2, "1 segment, fin centered"))
        n = self.shape.tail_segments
        last = S.b3 - self.p.tail_root_len
        rows.append(("tail", S.b2, last,
                     f"{n} segment{'s' if n > 1 else ''} of "
                     f"{(last - S.b2) / n:.1f} mm"))
        x1 = S.caudal[:, 0].max() if S.caudal is not None else S.b3
        rows.append(("caudal", last, x1, "tail root + fan piece"))
        return rows

    # ---------------- fins from the drawn outlines --------------------
    @staticmethod
    def _side_slice(X, Z):
        """Fin silhouettes live in (x,z); collapse the y axis of a full
        meshgrid so sd_polygon runs on nx*nz points, not nx*ny*nz."""
        if X.ndim == 3 and X.shape[1] > 1:
            return X[:, :1, :], Z[:, :1, :]
        return X, Z

    @staticmethod
    def _top_slice(X, Y):
        if X.ndim == 3 and X.shape[2] > 1:
            return X[:, :, :1], Y[:, :, :1]
        return X, Y

    def with_fins(self, X, Y, Z):
        p, S = self.p, self.shape
        d = self.core(X, Y, Z)

        if S.caudal is not None:
            Xs, Zs = self._side_slice(X, Z)
            c2 = sd_polygon(S.caudal[:, 0].astype(F32),
                            S.caudal[:, 1].astype(F32), Xs, Zs)
            d = smin(d, fin_sheet(c2, Y,
                                  p.tail_thick_root * S.caudal_thick,
                                  p.tail_thick_tip * S.caudal_thick,
                                  X, float(S.caudal[:, 0].min()),
                                  float(S.caudal[:, 0].max()),
                                  p.fin_edge / 2, p.fin_fillet), F32(6.0))

        if S.dorsal is not None:
            Xs, Zs = self._side_slice(X, Z)
            d2 = sd_polygon(S.dorsal[:, 0].astype(F32),
                            S.dorsal[:, 1].astype(F32), Xs, Zs)
            lo, hi = self.dorsal_trim
            d2 = smax(d2, (lo - Xs).astype(F32), F32(1.5))
            d2 = smax(d2, (Xs - hi).astype(F32), F32(1.5))
            d = smin(d, fin_sheet(d2, Y, p.dorsal_thickness,
                                  0.65 * p.dorsal_thickness, Z,
                                  S.dorsal_root_z, S.dorsal_top_z,
                                  p.fin_edge / 2, p.fin_fillet), F32(5.0))

        for g in self._fins():
            for sgn in (1.0, -1.0):
                boss = (np.sqrt((X - g["cx"]) ** 2 + (Y - sgn * g["cy"]) ** 2
                                + (Z - g["cz"]) ** 2) - g["boss_r"])
                d = smin(d, boss.astype(F32), F32(3.0))
        return d

    # ---------------- side fins: drawn paddles on ball joints ----------
    def _fins(self):
        p, out = self.p, []
        for g in self.shape.side_fins:
            rb = min(max(0.42 * g["chord"], 3.0), 5.0)
            zc = max(0.62 * rb, float(np.sqrt(2 * rb * p.clearance)) + 0.45)
            xb = float(g["att"][0])
            cy = self.halfwidth_at(xb, zc) - 0.2 * rb
            out.append(dict(cx=xb, cy=cy, cz=zc, rb=rb,
                            dx=float(g["dirv"][0]), dy=float(g["dirv"][1]),
                            span=g["span"], chord=g["chord"], th=g["thick"],
                            rn=0.45 * rb, boss_r=rb + p.clearance + p.wall,
                            poly=g["poly"], att=g["att"]))
        return out

    def fin_part(self, g, sgn, X, Y, Z):
        """Ball + neck (inherited design) + the drawn paddle, extruded."""
        p = self.p
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
        Xs, Ys = self._top_slice(X, Y)
        d2 = sd_polygon(g["poly"][:, 0].astype(F32),
                        (sgn * g["poly"][:, 1]).astype(F32), Xs, Ys)
        # trim where the paddle would run into the body, measured at the
        # paddle's TOP -- the body overhangs outward as z rises, so that
        # is where the two would touch first
        # 1.0 mm: enough that a fully flexed neighbour segment (its face
        # sweeps sideways by ~seg_overhang * tan(swing)) still clears
        body2d = self.core(Xs, Ys, np.full_like(Xs, g["th"]))
        d2 = smax(d2, (1.0 - body2d).astype(F32), F32(2.0))
        u = ((X - g["att"][0]) * g["dx"] + (Y - sgn * g["att"][1]) * dy)
        pad = fin_sheet(d2, Z - zf, g["th"] / 2, 0.75 * g["th"] / 2,
                        u, 0.0, max(g["span"], 1.0), 0.45, p.fin_fillet)
        # keep-out around the socket boss (only the neck may enter it)
        rball = np.sqrt((X - g["cx"]) ** 2 + (Y - cy) ** 2
                        + (Z - g["cz"]) ** 2)
        pad = smax(pad, ((g["boss_r"] + 0.7) - rball).astype(F32), F32(1.5))
        f = smin(smin(ball, neck, F32(1.5)), pad, F32(2.0))
        return np.maximum(f, (-Z).astype(F32))

    # ---------------- face: eyes/lips/mouth ----------------------------
    def _front_at(self, z):
        """x of the nose front surface at height z (y = 0)."""
        xs = np.arange(self.shape.b0 - 1.0, self.shape.b0 + 40.0, 0.05,
                       dtype=F32)
        f = self.core(xs, np.zeros_like(xs), np.full_like(xs, z))
        idx = np.nonzero(f < 0)[0]
        return float(xs[idx[0]]) if len(idx) else self.shape.b0 + 2.0

    def styled(self, X, Y, Z):
        d = super().styled(X, Y, Z)
        S = self.shape
        m = S.mouth
        if m.get("shape") in ("groove", "curve"):
            # a groove carved where the mouth cut surface meets the nose:
            # a tilted plane through the nose (groove) or the drawn side-
            # view curve swept across it (curve). Limited to `length` mm
            # around the arc midpoint, never below z = 2 (build plate + 2)
            tf = self.top_at(0.05 * self.L)
            if m["shape"] == "groove" or S.mouth_curve is None:
                zm = max(0.34 * tf + m["height"], 4.0)
                xc = self._front_at(zm)
                a = np.deg2rad(m["tilt"])
                t2d = np.abs((X - xc) * np.cos(a) + (Z - zm) * np.sin(a))
                cx, cz = xc, zm
            else:
                t2d = _polyline_dist(X, Z, S.mouth_curve)
                cx, cz = S.mouth_curve[len(S.mouth_curve) // 2]
            reach = (np.sqrt((X - cx) ** 2 + Y ** 2 + (Z - cz) ** 2)
                     - m["length"] / 2)
            g = np.maximum(t2d - 0.8, reach)
            g = np.maximum(g, 2.0 - Z)          # stay 2 mm off the plate
            g = np.maximum(g, -(d + 1.4))       # only carve near the skin
            d = smax(d, -g.astype(F32), F32(0.6))
        if S.mouth_open > 0:
            tf = self.top_at(S.b0 + 0.05 * (S.b3 - S.b0))
            R = np.clip(S.mouth_open, 0.0, 1.0) * 0.38 * tf
            pocket = sd_ellipsoid(X, Y, Z, S.b0 + 0.25 * R, 0.0, 0.36 * tf,
                                  R, 0.85 * R, 0.55 * R)
            d = smax(d, -pocket, F32(1.2))
        return d

    def bounds(self):
        p, S = self.p, self.shape
        x1 = S.b3
        z1 = float(S.ztop.max())
        if S.caudal is not None:
            x1 = max(x1, float(S.caudal[:, 0].max()))
            z1 = max(z1, float(S.caudal[:, 1].max()))
        if S.dorsal is not None:
            z1 = max(z1, S.dorsal_top_z)
        yw = float(S.wid.max()) + p.eye_proud
        for g in S.side_fins:
            yw = max(yw, float(g["poly"][:, 1].max()))
        return S.b0 - 3.0, x1 + 4.0, -yw - 3.0, yw + 3.0, -1.2, z1 + 3.0


# ======================================================================
# SVG template: the two drawing views with regions and joint cuts
# ======================================================================
def write_svg(path, builder: NurbsFishBuilder, spec: dict):
    S = builder.shape
    merged = deep_merge(DEFAULT_SHAPE, spec or {})
    x0, x1 = S.b0 - 12, (S.caudal[:, 0].max() if S.caudal is not None
                         else S.b3) + 12
    W = x1 - x0
    side_h, top_h = 72, 52
    pad, gap = 10, 16
    H = side_h + top_h + 2 * pad + gap
    e = []

    def sx(x):
        return x - x0 + pad

    def sz(z):
        return pad + side_h - z          # side view: z up

    def sy(y):
        return pad + side_h + gap + top_h / 2 - y   # top view: centerline mid

    def poly(pts, fy, cls, close=False):
        d = " ".join(f"{sx(x):.1f},{fy(v):.1f}" for x, v in pts)
        tag = "polygon" if close else "polyline"
        e.append(f'<{tag} class="{cls}" points="{d}"/>')

    for name, (view, pts) in S.svg_curves.items():
        fy = sz if view == "side" else sy
        closed = name.endswith("_fin")
        poly(pts, fy, "fin" if closed else "body", closed)
        if view == "top" and not closed:
            poly(np.column_stack([pts[:, 0], -pts[:, 1]]), sy, "body")
        if view == "top" and closed:
            poly(np.column_stack([pts[:, 0], -pts[:, 1]]), sy, "fin mirror",
                 True)
    for name, cur in merged.get("curves", {}).items():
        view = ("top" if name in ("plan", "pectoral_fin", "pelvic_fin")
                else "side")
        fy = sz if view == "side" else sy
        cps = np.asarray(cur["points"], dtype=float)
        poly(cps, fy, "cage", cur.get("closed", False))
        for x, v in cps:
            e.append(f'<circle class="cp" cx="{sx(x):.1f}" '
                     f'cy="{fy(v):.1f}" r="1.1"/>')

    labels = {"head": (S.b0, S.b1), "dorsal": (S.b1, S.b2),
              "tail": (S.b2, S.b3 - builder.p.tail_root_len),
              "caudal": (S.b3 - builder.p.tail_root_len, x1 - 12)}
    if builder.dorsal_on_head:
        labels.pop("dorsal")
        labels["head"] = (S.b0, S.b2)
    for name, (a, b) in labels.items():
        e.append(f'<text class="lbl" x="{sx((a + b) / 2):.1f}" '
                 f'y="{pad + 5:.1f}">{name}</text>')
        e.append(f'<line class="region" x1="{sx(b):.1f}" y1="{pad:.1f}" '
                 f'x2="{sx(b):.1f}" y2="{H - pad:.1f}"/>')
    for c in builder.cuts:
        e.append(f'<line class="cut" x1="{sx(c):.1f}" y1="{pad + 8:.1f}" '
                 f'x2="{sx(c):.1f}" y2="{H - pad:.1f}"/>')
    e.append(f'<line class="plate" x1="{pad}" y1="{sz(0):.1f}" '
             f'x2="{W + pad:.1f}" y2="{sz(0):.1f}"/>')
    e.append(f'<line class="plate" x1="{pad}" y1="{sy(0):.1f}" '
             f'x2="{W + pad:.1f}" y2="{sy(0):.1f}"/>')
    e.append(f'<text class="lbl" x="{pad + 2}" y="{sz(0) - 2:.1f}">side '
             f'(z=0 build plate)</text>')
    e.append(f'<text class="lbl" x="{pad + 2}" y="{sy(0) - 2:.1f}">top '
             f'(centerline)</text>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {W + 2 * pad:.0f} {H:.0f}" '
           f'width="{(W + 2 * pad) * 6:.0f}" height="{H * 6:.0f}">'
           '<style>'
           '.body{fill:none;stroke:#1a6faf;stroke-width:.8}'
           '.fin{fill:#1a6faf22;stroke:#1a6faf;stroke-width:.6}'
           '.mirror{opacity:.35}'
           '.cage{fill:none;stroke:#c96;stroke-width:.35;stroke-dasharray:1.2 1.2}'
           '.cp{fill:#c96}'
           '.region{stroke:#a33;stroke-width:.5;stroke-dasharray:3 2}'
           '.cut{stroke:#888;stroke-width:.35;stroke-dasharray:.8 1.6}'
           '.plate{stroke:#494;stroke-width:.4}'
           '.lbl{font:4.5px sans-serif;fill:#a33}'
           'svg{background:#fff}</style>'
           f'<rect width="100%" height="100%" fill="white"/>{"".join(e)}</svg>')
    with open(path, "w") as f:
        f.write(svg)


# ======================================================================
# CLI
# ======================================================================
def apply_sets(spec: dict, sets: list[str]) -> dict:
    """--set head.height=1.2 / --set regions.tail_segments=4 into the spec."""
    for s in sets:
        try:
            path, val = s.split("=", 1)
            keys = path.strip().split(".")
            val = json.loads(val)
        except ValueError:
            raise SystemExit(f"bad --set '{s}' (want e.g. tail.length=1.3)")
        if keys[0] not in ("regions", "curves", "sliders", "mouth"):
            keys = ["sliders"] + keys
        node = spec
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="NURBS-drawn flexi fish (regions + sliders)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="run --dump-shape for an editable template of every curve")
    ap.add_argument("--shape", help="JSON shape file (curves/regions/sliders); "
                                    "deep-merged over the built-in default")
    ap.add_argument("--config", help="JSON overriding FishParams (joints, "
                                     "eyes, walls -- see flexifish.py)")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="tweak a slider/region inline, e.g. tail.length=1.3, "
                         "regions.tail_segments=4 (repeatable)")
    ap.add_argument("--out", default="fish_nurbs.stl")
    ap.add_argument("--res", type=float, help="voxel size override (mm)")
    ap.add_argument("--preview", action="store_true",
                    help="fast coarse build (use full res for printing)")
    ap.add_argument("--coupon", action="store_true",
                    help="also write a one-joint test print")
    ap.add_argument("--png", action="store_true", help="render a 4-view PNG")
    ap.add_argument("--svg", action="store_true",
                    help="write a 2-view curve template SVG (side + top, "
                         "with regions, joint cuts and control cages)")
    ap.add_argument("--dump-shape", action="store_true",
                    help="write the default shape to fish_shape_default.json")
    ap.add_argument("--dump-config", action="store_true")
    ap.add_argument("--list-params", action="store_true")
    args = ap.parse_args(argv)

    if args.dump_shape:
        json.dump(DEFAULT_SHAPE, open("fish_shape_default.json", "w"),
                  indent=1)
        print("wrote fish_shape_default.json -- edit curves, then "
              "--shape fish_shape_default.json")
        return
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
            sys.exit(f"unknown parameter(s): {sorted(unknown)}")
        p = replace(p, **overrides)
    if args.res:
        p = replace(p, res=args.res)
    res = 0.62 if args.preview and not args.res else p.res

    spec = json.load(open(args.shape)) if args.shape else {}
    spec = apply_sets(spec, args.set)

    t0 = time.time()
    b = NurbsFishBuilder(p, spec)
    for w in b.warnings:
        print("note:", w)
    print("regions:")
    for name, a, x1, what in b.region_table():
        print(f"  {name:10s} {a:6.1f} -> {x1:6.1f} mm  {what}")
    jxs = ", ".join(f"{j['xa']:.0f}" for j in b.joints)
    print(f"fish: {b.bounds()[1] - b.bounds()[0]:.0f} mm long, "
          f"{b.p.n_segments} segments, {b.p.joint_style} joints at x = {jxs}")

    if args.svg:
        svg = args.out.replace(".stl", "") + "_curves.svg"
        write_svg(svg, b, spec)
        print(f"wrote {svg}")

    verts, faces = mesh(b, res)
    write_stl(args.out, verts, faces)
    man, shells = mesh_stats(verts, faces)
    expected = b.p.n_segments + 2 + 2 * len(b.shape.side_fins)
    print(f"{args.out}: {len(faces)} tris, manifold={man}, "
          f"shells={shells} (expected {expected})  [{time.time() - t0:.0f}s]")
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
