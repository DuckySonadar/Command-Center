#!/usr/bin/env python3
"""
bg_remove.py — cut objects out of a predominantly uniform-color background.

Pipeline
--------
1. Estimate the background color by sampling the image border (or take
   --bg-color). Work in CIE Lab space so "color distance" matches what
   your eye considers different.
2. Build a foreground mask: every pixel whose Lab distance from the
   background exceeds --tolerance.
3. Clean the mask: morphological open/close, drop small specks
   (--min-area), optionally fill enclosed holes.
4. Optional border prediction (--predict): walk each object's contour,
   score how *distinct* each boundary point is (how far the colors just
   inside the edge sit from the background). Contiguous runs of
   low-confidence points — places where the object fades into the
   background — are replaced by a cubic Hermite spline anchored on the
   distinct segments at either side, using their tangent directions to
   approximate what the hidden border probably looks like.
5. Alpha: hard cut, or soft color ramp near the edge (--soft), or
   geometric feather (--feather). Output is a PNG with transparency.

Try it without your own image:
    python3 bg_remove.py --demo demo.png
    python3 bg_remove.py demo.png cutout.png --predict --debug
"""

import argparse
import os
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Image loading (incl. HEIC/HEIF), background estimation & color distance
# --------------------------------------------------------------------------

def load_image(path):
    """Read an image as uint8 BGR. Falls back to Pillow (with the
    pillow-heif plugin when present) for formats OpenCV can't read,
    notably iPhone HEIC/HEIF files."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        sys.exit(f"error: could not read {path} with OpenCV, and Pillow "
                 "is not installed. For HEIC/HEIF support run: "
                 "pip install pillow pillow-heif")
    except Exception as e:
        sys.exit(f"error: could not read {path}: {e}\n"
                 "(for HEIC/HEIF support: pip install pillow pillow-heif)")


def to_lab(bgr):
    """uint8 BGR -> float32 Lab (OpenCV scaling, all channels 0..255)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _subsample(samples, cap):
    if len(samples) > cap:  # k-means doesn't need every pixel
        idx = np.random.default_rng(0).choice(len(samples), cap, replace=False)
        samples = samples[idx]
    return samples


def bg_samples(lab, bg_from="border", border_frac=0.04):
    """Pixels believed to be background, per the --bg-from strategy.

    border  -- a frame around the image edge (default; assumes nothing
               but backdrop touches the frame)
    image   -- the whole image, subsampled (assumes the backdrop is the
               dominant color overall — survives clutter at the edges)
    "fx,fy" -- a patch around that point, in 0-1 fractional coordinates
               (point at a clean piece of backdrop when both assumptions
               fail, e.g. "0.5,0.9" = bottom middle)
    """
    h, w = lab.shape[:2]
    if bg_from == "border":
        b = max(2, int(round(border_frac * min(h, w))))
        strips = [lab[:b, :], lab[-b:, :], lab[:, :b], lab[:, -b:]]
        samples = np.concatenate([s.reshape(-1, 3) for s in strips])
    elif bg_from == "image":
        samples = lab.reshape(-1, 3)
    else:
        try:
            fx, fy = (float(v) for v in bg_from.split(","))
        except ValueError:
            sys.exit(f"error: --bg-from must be 'border', 'image', or "
                     f"'fx,fy' fractional coordinates, not {bg_from!r}")
        r = max(4, int(round(0.03 * min(h, w))))
        cx, cy = int(fx * (w - 1)), int(fy * (h - 1))
        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)
        samples = lab[y0:y1, x0:x1].reshape(-1, 3)
    return _subsample(samples.astype(np.float32), 20000)


def estimate_bg_colors(samples, max_colors=1, min_share=0.08):
    """Dominant Lab color(s) of the background sample set.

    With max_colors=1 you get the single biggest k-means cluster; higher
    values also keep secondary clusters that cover at least `min_share`
    of the samples — useful when the backdrop has a visible gradient,
    vignette, or two-toned lighting, so the "background" is really a
    range of colors. K-means (rather than a plain mean) keeps an object
    or clutter in the sample region from skewing the estimate.
    """
    k = max(3, max_colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(counts)[::-1]
    kept = [centers[order[0]]]
    for i in order[1:max_colors]:
        if counts[i] >= min_share * len(samples):
            kept.append(centers[i])
    return np.array(kept, np.float32)


def color_distance(lab, bg_colors, lightness_weight=1.0, sat_weight=1.0):
    """Per-pixel weighted Lab distance to the *nearest* background color.

    lightness_weight scales the L channel: <1 tolerates shadows and
    brightness gradients, >1 makes brightness differences count more.
    sat_weight scales the a/b (chroma) channels: >1 makes saturation and
    hue differences count more — handy when the object differs from the
    backdrop mainly in colorfulness rather than brightness.
    """
    bg_colors = np.asarray(bg_colors, np.float32).reshape(-1, 3)
    weights = np.array([lightness_weight, sat_weight, sat_weight],
                       np.float32).reshape(1, 1, 3)
    dist = None
    for bg in bg_colors:
        diff = (lab - bg.reshape(1, 1, 3)) * weights
        d = np.sqrt((diff ** 2).sum(axis=2))
        dist = d if dist is None else np.minimum(dist, d)
    return dist


def auto_tolerance(samples, bg_colors, lightness_weight, sat_weight,
                   trim=0.75, margin=1.5, floor=10.0):
    """Pick a tolerance from the background samples themselves: how far
    do genuine background pixels stray from the background model?

    Only the closest `trim` fraction of samples is considered, so
    clutter that slipped into the sample region (a keyboard at the edge
    of frame, glare) inflates neither the estimate nor your patience.
    The threshold sits `margin` above that trimmed tail."""
    d = color_distance(samples.reshape(1, -1, 3), bg_colors,
                       lightness_weight, sat_weight).ravel()
    kept = np.sort(d)[:max(1, int(len(d) * trim))]
    return max(float(np.percentile(kept, 99.0)) * margin, floor)


# --------------------------------------------------------------------------
# Mask building & cleanup
# --------------------------------------------------------------------------

def build_mask(dist, tolerance):
    return (dist > tolerance).astype(np.uint8) * 255


def clean_mask(mask, open_size=3, close_size=5, min_area=400, fill_holes=True,
               keep_largest=0):
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 0 or keep_largest > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        allowed = set(range(1, n))
        if keep_largest > 0:  # e.g. 1 object photographed on a busy desk
            biggest = np.argsort(areas)[::-1][:keep_largest] + 1
            allowed &= set(int(i) for i in biggest)
        keep = np.zeros_like(mask)
        for i in allowed:
            if areas[i - 1] >= min_area:
                keep[labels == i] = 255
        mask = keep

    if fill_holes:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    return mask


U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"


def ai_alpha(img):
    """Alpha matte from the U^2-Net neural model, 0..1 float.

    Uses the rembg package when installed; otherwise falls back to a
    built-in runner that only needs `pip install onnxruntime` (rembg
    drags in numba/llvmlite, which don't build everywhere). Either way
    the first run downloads the model (~170 MB) to ~/.u2net and
    everything runs locally. Use when the photo defeats color logic —
    e.g. glare on a glossy surface that shares the object's color."""
    try:
        from rembg import remove, new_session
    except ImportError:
        return _u2net_alpha(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = np.asarray(remove(rgb, session=new_session("u2net")))
    return out[..., 3].astype(np.float32) / 255.0


def _download(url, dest):
    """Fetch url to dest with a progress line. Uses certifi's CA bundle
    when available — python.org installs on macOS ship without CA
    certificates wired up, and the fix is friendlier than a traceback."""
    import ssl
    import urllib.error
    import urllib.request

    def fetch(ctx):
        with urllib.request.urlopen(url, context=ctx) as r, \
                open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB",
                          end="", flush=True)
            print()

    contexts = [None]  # the system trust store is usually right; try first
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    for ctx in contexts:
        try:
            fetch(ctx)
            return
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
    sys.exit(
        "error: your Python can't verify HTTPS certificates, so the "
        "model download failed.\nOn macOS (python.org installs) run "
        "the bundled fixer once:\n"
        '    open "/Applications/Python 3.13/Install Certificates.command"\n'
        "(adjust the version to yours), or: pip3 install certifi\n"
        "Then re-run this command.")


def _u2net_model_path():
    home = os.path.expanduser(
        os.getenv("U2NET_HOME",
                  os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")))
    path = os.path.join(home, "u2net.onnx")
    if not os.path.exists(path):
        os.makedirs(home, exist_ok=True)
        print(f"downloading U^2-Net model (~170 MB) to {path} ...")
        tmp = path + ".part"
        _download(U2NET_URL, tmp)
        import hashlib
        md5 = hashlib.md5()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
        if md5.hexdigest() != U2NET_MD5:
            os.remove(tmp)
            sys.exit("error: model download was corrupted — please retry")
        os.replace(tmp, path)
    return path


def _u2net_alpha(img):
    """Minimal U^2-Net inference on plain onnxruntime; mirrors rembg's
    pre/post-processing (320x320, ImageNet-style normalization over the
    image max, min-max rescale of the predicted matte)."""
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("error: --ai needs onnxruntime: pip install onnxruntime "
                 "(or the full rembg package, where it builds)")
    sess = ort.InferenceSession(_u2net_model_path(),
                                providers=["CPUExecutionProvider"])
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (320, 320),
                       interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    small /= max(float(small.max()), 1e-6)
    small = (small - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    inp = small.transpose(2, 0, 1)[None].astype(np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    alpha = cv2.resize(pred.astype(np.float32), (w, h),
                       interpolation=cv2.INTER_LANCZOS4)
    return np.clip(alpha, 0.0, 1.0)


def grabcut_refine(img, mask, iters=3, max_dim=1400):
    """Refine a color-threshold mask with OpenCV's GrabCut.

    The threshold mask seeds GrabCut's labels: strongly-inside pixels
    become sure-foreground, the rest of the mask probable-foreground, a
    band around it probable-background, and everything far away
    sure-background. GrabCut then fits Gaussian mixture color models to
    each side and solves a graph cut, which recovers objects that share
    *some* colors with the backdrop (glare, sheen) because it also
    weighs spatial coherence. Runs on a downscaled copy for speed; the
    result is upscaled back.
    """
    h, w = mask.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(mask, (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    else:
        small, msmall = img, mask

    def kernel(frac):
        k = max(3, int(frac * min(msmall.shape)) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    gc = np.full(msmall.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(msmall, kernel(0.10)) == 0] = cv2.GC_BGD
    gc[msmall > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(msmall, kernel(0.02)) > 0] = cv2.GC_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0)
    out = out.astype(np.uint8)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# --------------------------------------------------------------------------
# Tangent-based border prediction for indistinct edge regions
# --------------------------------------------------------------------------

def _smooth_closed(points, k):
    """Circular moving-average smoothing of an (N,2) closed polyline."""
    if k <= 1:
        return points.astype(np.float64)
    kernel = np.ones(k) / k
    out = np.empty_like(points, dtype=np.float64)
    for c in range(2):
        col = points[:, c].astype(np.float64)
        padded = np.concatenate([col[-k:], col, col[:k]])
        out[:, c] = np.convolve(padded, kernel, mode="same")[k:-k]
    return out


def _tangents_closed(points):
    """Unit tangents of a closed polyline via central differences."""
    t = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return t / norms


def boundary_confidence(contour, dist_map, mask, sample_depth=4):
    """How distinct each contour point's edge is.

    For each point, probe `sample_depth` px along the normal on both
    sides of the edge and measure the *contrast*: color distance from
    the background just inside minus just outside. A crisp edge jumps
    from near-zero (background) to a large value (object) and scores
    high; where the object fades into the background both probes read
    similar mid values and the contrast collapses. Contrast is used
    rather than the raw inside distance because pixels adjacent to a
    threshold-derived contour always sit near the tolerance, which would
    make every edge look equally (in)distinct.
    Returned values are Lab-distance units (compare against tolerance).
    """
    h, w = dist_map.shape
    pts = contour.astype(np.float64)
    smooth = _smooth_closed(pts, 7)
    tan = _tangents_closed(smooth)
    normal = np.stack([-tan[:, 1], tan[:, 0]], axis=1)  # 90° rotation

    def sample(offsets):
        p = np.clip(pts + offsets, 0, [w - 1, h - 1]).round().astype(int)
        return dist_map[p[:, 1], p[:, 0]], mask[p[:, 1], p[:, 0]] > 0

    # The normal's sign is arbitrary; probe both sides and call the side
    # that lands on foreground "inside" (per point, so concavities work).
    da, a_inside = sample(normal * sample_depth)
    db, _ = sample(-normal * sample_depth)
    d_in = np.where(a_inside, da, db)
    d_out = np.where(a_inside, db, da)
    return d_in - d_out


def _hermite(p0, p1, m0, m1, n):
    """Cubic Hermite curve from p0 to p1 with tangent directions m0, m1."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    scale = np.linalg.norm(p1 - p0)  # tangent magnitude ~ chord length
    return h00 * p0 + h10 * m0 * scale + h01 * p1 + h11 * m1 * scale


def _low_conf_runs(low):
    """Contiguous runs of True in a circular boolean array -> (start, len)."""
    n = len(low)
    if low.all() or not low.any():
        return []
    # rotate so index 0 is distinct, then runs never wrap
    shift = int(np.argmin(low))
    rolled = np.roll(low, -shift)
    runs, i = [], 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            runs.append(((i + shift) % n, j - i))
            i = j
        else:
            i += 1
    return runs


def _consolidate(low, min_gap, min_island):
    """Denoise the per-point indistinct flags.

    Two low runs separated by fewer distinct points than an anchor needs
    are one gap — a tiny distinct island can't provide a trustworthy
    tangent, so merge across it. Then drop low specks shorter than
    min_gap, so stray points can't poison the anchor windows either.
    """
    out = low.copy()
    for s, l in _low_conf_runs(~out):        # short distinct islands -> low
        if l < min_island:
            out[np.arange(s, s + l) % len(out)] = True
    for s, l in _low_conf_runs(out):         # short low specks -> distinct
        if l < min_gap:
            out[np.arange(s, s + l) % len(out)] = False
    return out


def predict_borders(mask, dist_map, tolerance, conf_ratio=1.3,
                    max_gap_frac=0.35, min_gap_px=8, anchor_pts=12,
                    sample_depth=4, debug_img=None):
    """Replace indistinct contour stretches with tangent-extrapolated curves.

    conf_ratio    -- a boundary point is "indistinct" when its edge
                     contrast is below tolerance * conf_ratio.
    max_gap_frac  -- longest gap we dare to bridge, as fraction of the
                     contour perimeter (bridging half an object is fiction).
    min_gap_px    -- ignore gaps shorter than this; cleanup handles those.
    anchor_pts    -- how many distinct points on each side define the
                     anchor position and tangent direction.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    new_mask = np.zeros_like(mask)
    n_bridged = 0

    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)  # (N, 2) x,y
        n = len(pts)
        if n < 4 * anchor_pts:
            cv2.drawContours(new_mask, [cnt], -1, 255, cv2.FILLED)
            continue

        conf = boundary_confidence(pts, dist_map, mask, sample_depth)
        low = _consolidate(conf < tolerance * conf_ratio, min_gap_px,
                           min_island=2 * anchor_pts)
        smooth = _smooth_closed(pts, 7)
        tans = _tangents_closed(smooth)

        segments = []  # (start_index, replacement_points or None)
        for start, length in _low_conf_runs(low):
            if length < min_gap_px or length > max_gap_frac * n:
                continue
            # anchors: last distinct point before the run / first after
            i0 = (start - 1) % n
            i1 = (start + length) % n
            pre = [(i0 - k) % n for k in range(anchor_pts)]
            post = [(i1 + k) % n for k in range(anchor_pts)]
            if low[pre].any() or low[post].any():
                continue  # neighbors not solidly distinct; don't guess
            p0, p1 = smooth[i0], smooth[i1]
            m0 = np.mean(tans[pre], axis=0)   # direction of travel into gap
            m1 = np.mean(tans[post], axis=0)  # direction of travel out of it
            for m in (m0, m1):
                nm = np.linalg.norm(m)
                if nm > 1e-9:
                    m /= nm
            curve = _hermite(p0, p1, m0, m1, max(length, 8))
            segments.append((start, length, curve))
            n_bridged += 1

        if debug_img is not None:
            for j in range(n):
                x, y = pts[j].astype(int)
                color = (0, 0, 255) if low[j] else (0, 255, 0)
                cv2.circle(debug_img, (x, y), 1, color, -1)
            for _, _, curve in segments:
                c = curve.round().astype(np.int32)
                cv2.polylines(debug_img, [c], False, (255, 0, 255), 2)

        if not segments:
            cv2.drawContours(new_mask, [cnt], -1, 255, cv2.FILLED)
            continue

        # stitch: original points outside gaps + spline points inside them
        replaced = np.zeros(n, dtype=bool)
        insert_at = {}
        for start, length, curve in segments:
            for k in range(length):
                replaced[(start + k) % n] = True
            insert_at[start] = curve

        out = []
        i = 0
        visited = 0
        while visited < n:
            if i in insert_at:
                out.extend(insert_at[i])
                run_len = next(l for s, l, _ in segments if s == i)
                i = (i + run_len) % n
                visited += run_len
            else:
                out.append(pts[i])
                i = (i + 1) % n
                visited += 1
        poly = np.array(out).round().astype(np.int32)
        cv2.fillPoly(new_mask, [poly], 255)

    return new_mask, n_bridged


# --------------------------------------------------------------------------
# Alpha shaping
# --------------------------------------------------------------------------

def make_alpha(mask, dist_map, tolerance, soft=0.0, soft_band=6, feather=0.0):
    """Turn a binary mask into an alpha channel.

    soft    -- width (in Lab color-distance units) of a linear ramp around
               the tolerance, applied only within `soft_band` px of the
               mask edge, so interiors — including predicted regions where
               the color signal is weak — stay fully opaque.
    feather -- Gaussian blur sigma on the final alpha (geometric soften;
               works everywhere, including predicted borders).
    """
    alpha = (mask > 0).astype(np.float32)

    if soft > 0:
        lo, hi = tolerance - soft, tolerance + soft
        ramp = np.clip((dist_map - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        band_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * soft_band + 1, 2 * soft_band + 1))
        inner = cv2.erode(mask, band_k) > 0
        outer = cv2.dilate(mask, band_k) > 0
        band = outer & ~inner
        alpha[band] = ramp[band]
        alpha[inner] = 1.0
        alpha[~outer] = 0.0

    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha *= (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0)

    return np.clip(alpha, 0.0, 1.0)


# --------------------------------------------------------------------------
# Demo image
# --------------------------------------------------------------------------

def make_demo(path):
    """Synthetic test image: uniform background, colorful object, and one
    region of the object deliberately faded into the background so the
    --predict feature has something to reconstruct."""
    h, w = 480, 640
    rng = np.random.default_rng(42)
    bg = np.full((h, w, 3), (188, 192, 190), np.uint8)  # light gray-green
    noise = rng.normal(0, 3, (h, w, 3))
    img = np.clip(bg + noise, 0, 255).astype(np.uint8)

    # colorful blob (ellipse with color bands)
    obj = np.zeros((h, w), np.uint8)
    cv2.ellipse(obj, (320, 240), (150, 110), 20, 0, 360, 255, -1)
    ys, xs = np.nonzero(obj)
    band = ((xs + ys) // 40) % 4
    palette = np.array([(40, 60, 200), (30, 160, 240),
                        (60, 170, 70), (170, 90, 40)], np.uint8)
    img[ys, xs] = palette[band]

    # fade one arc of the boundary into the background (the "indistinct" area)
    fade = np.zeros((h, w), np.float32)
    cv2.ellipse(fade, (320, 240), (150, 110), 20, -30, 50, 1.0, 60)
    fade = cv2.GaussianBlur(fade, (0, 0), 15) * (obj > 0)
    fade = np.clip(fade * 1.6, 0, 1)[..., None]
    img = (img * (1 - fade) + bg * fade).astype(np.uint8)

    cv2.imwrite(path, img)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input image (or output path with --demo)")
    ap.add_argument("output", nargs="?", help="output PNG with alpha "
                    "(default: <input>_cutout.png)")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic test image at INPUT and exit")

    g = ap.add_argument_group("background & threshold")
    g.add_argument("--tolerance", default="28",
                   help="Lab color distance from background that counts as "
                        "foreground (default 28; lower keeps more), or "
                        "'auto' to derive it from the border pixels")
    g.add_argument("--bg-color", metavar="R,G,B[;R,G,B...]",
                   help="background color override; skips auto-estimation. "
                        "Separate several colors with ';' to model a "
                        "multi-toned backdrop")
    g.add_argument("--bg-colors", type=int, default=1, metavar="N",
                   help="model the background with up to N auto-estimated "
                        "colors instead of 1 — use 2-4 for backdrops with "
                        "gradients or uneven lighting (default 1)")
    g.add_argument("--bg-from", default="border", metavar="WHERE",
                   help="where to sample the background from: 'border' "
                        "(default), 'image' (dominant color of the whole "
                        "frame — use when clutter touches the edges), or "
                        "'fx,fy' fractional coordinates of a clean backdrop "
                        "spot, e.g. '0.5,0.9'")
    g.add_argument("--border-frac", type=float, default=0.04,
                   help="border width sampled for bg estimation, as a "
                        "fraction of image size (default 0.04)")
    g.add_argument("--lightness-weight", type=float, default=1.0,
                   help="weight of the L (brightness) channel in color "
                        "distance; <1 tolerates shadows/gradients, >1 "
                        "emphasizes brightness (default 1)")
    g.add_argument("--sat-weight", type=float, default=1.0,
                   help="weight of the a/b (saturation & hue) channels in "
                        "color distance; >1 separates objects that differ "
                        "from the backdrop mainly in colorfulness (default 1)")

    g = ap.add_argument_group("mask cleanup")
    g.add_argument("--open", type=int, default=3, dest="open_size",
                   help="morphological open kernel px, removes specks (default 3)")
    g.add_argument("--close", type=int, default=5, dest="close_size",
                   help="morphological close kernel px, seals pinholes (default 5)")
    g.add_argument("--min-area", type=int, default=400,
                   help="drop connected components smaller than this (default 400)")
    g.add_argument("--keep-holes", action="store_true",
                   help="keep background-colored holes inside objects "
                        "transparent (default: holes are filled)")
    g.add_argument("--keep-largest", type=int, default=0, metavar="N",
                   help="keep only the N largest objects — drops desk "
                        "clutter that isn't background-colored (default 0 "
                        "= keep everything above --min-area)")
    g.add_argument("--grabcut", type=int, default=0, metavar="ITERS",
                   help="refine the mask with GrabCut for this many "
                        "iterations (try 3). Fits color-mixture models to "
                        "object and backdrop with spatial smoothness — "
                        "rescues shots where glare or sheen shares colors "
                        "with the object, at the cost of a few seconds")
    g.add_argument("--roi", metavar="FX,FY,FW,FH",
                   help="only look for objects inside this rectangle "
                        "(fractional: x,y,width,height, e.g. "
                        "'0.1,0.2,0.8,0.7'); everything outside becomes "
                        "transparent. Use when the frame edge has clutter "
                        "that isn't backdrop (desk, laptop, other items)")

    g = ap.add_argument_group("border prediction (tangent extrapolation)")
    g.add_argument("--predict", action="store_true",
                   help="bridge indistinct border stretches with Hermite "
                        "curves built from neighboring tangents")
    g.add_argument("--conf-ratio", type=float, default=1.3,
                   help="border point is 'indistinct' when its edge "
                        "contrast < tolerance * this (default 1.3)")
    g.add_argument("--max-gap-frac", type=float, default=0.35,
                   help="longest bridgeable gap as fraction of contour "
                        "perimeter (default 0.35)")
    g.add_argument("--min-gap", type=int, default=8,
                   help="ignore indistinct runs shorter than this many "
                        "contour px (default 8)")
    g.add_argument("--anchor-pts", type=int, default=12,
                   help="distinct contour points per side used to measure "
                        "the tangent (default 12)")
    g.add_argument("--sample-depth", type=int, default=4,
                   help="px stepped inside the edge when scoring "
                        "distinctness (default 4)")

    g = ap.add_argument_group("AI engine")
    g.add_argument("--ai", action="store_true",
                   help="segment with the rembg neural model (U^2-Net) "
                        "instead of color logic. Needs 'pip install rembg "
                        "onnxruntime'; first run downloads ~170 MB of model "
                        "weights. For photos color logic can't handle: "
                        "glare or reflections sharing the object's color. "
                        "--roi/--min-area/--keep-largest/--keep-holes/"
                        "--feather still apply; color options don't")

    g = ap.add_argument_group("edges & output")
    g.add_argument("--soft", type=float, default=0.0,
                   help="soft color ramp half-width in Lab units around the "
                        "tolerance (0 = hard edge)")
    g.add_argument("--soft-band", type=int, default=6,
                   help="px around the mask edge where --soft applies (default 6)")
    g.add_argument("--feather", type=float, default=1.0,
                   help="Gaussian sigma blur of the alpha edge (default 1.0, "
                        "0 = off)")
    g.add_argument("--debug", action="store_true",
                   help="also write *_mask.png, *_dist.png and, with "
                        "--predict, *_borders.png (green=distinct, "
                        "red=indistinct, magenta=predicted curve)")

    args = ap.parse_args()

    if args.demo:
        make_demo(args.input)
        print(f"demo image written to {args.input}")
        print(f"try: python3 {os.path.basename(sys.argv[0])} {args.input} "
              f"cutout.png --predict --debug")
        return

    img = load_image(args.input)
    out_path = args.output or os.path.splitext(args.input)[0] + "_cutout.png"
    stem = os.path.splitext(out_path)[0]

    full_img, roi_box = img, None
    if args.roi:
        try:
            fx, fy, fw, fh = (float(v) for v in args.roi.split(","))
        except ValueError:
            sys.exit(f"error: --roi must be 'fx,fy,fw,fh', not {args.roi!r}")
        H, W = img.shape[:2]
        x0, y0 = max(0, int(fx * W)), max(0, int(fy * H))
        x1, y1 = min(W, int((fx + fw) * W)), min(H, int((fy + fh) * H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            sys.exit("error: --roi rectangle is empty or out of frame")
        roi_box = (x0, y0, x1, y1)
        img = img[y0:y1, x0:x1]

    if args.ai:
        alpha, dist, tolerance = run_ai(img, args)
        mask = (alpha >= 0.5).astype(np.uint8) * 255
        if args.predict:
            print("note: --predict applies to the color pipeline; "
                  "ignored with --ai")
    else:
        alpha, mask, dist, tolerance = run_color(img, args, stem)

    if roi_box is not None:  # place the ROI's alpha back into the full frame
        x0, y0, x1, y1 = roi_box
        full_alpha = np.zeros(full_img.shape[:2], np.float32)
        full_alpha[y0:y1, x0:x1] = alpha
        img, alpha = full_img, full_alpha

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = (alpha * 255).round().astype(np.uint8)
    cv2.imwrite(out_path, rgba)
    kept = (alpha > 0.5).mean() * 100
    print(f"wrote {out_path}  ({kept:.1f}% of pixels kept)")

    if args.debug:
        cv2.imwrite(f"{stem}_mask.png", mask)
        extra = ""
        if dist is not None:
            d = np.clip(dist / max(tolerance * 2, 1e-6) * 255, 0, 255)
            cv2.imwrite(f"{stem}_dist.png", d.astype(np.uint8))
            extra = f", {stem}_dist.png"
        # cutout composited over a checkerboard — easiest way to eyeball it
        h, w = alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        checker = np.where(((xx // 16 + yy // 16) % 2) == 0, 200, 150)
        checker = np.repeat(checker[..., None], 3, axis=2).astype(np.float32)
        a = alpha[..., None]
        comp = (img.astype(np.float32) * a + checker * (1 - a))
        cv2.imwrite(f"{stem}_preview.png", comp.astype(np.uint8))
        print(f"debug images: {stem}_mask.png{extra}, {stem}_preview.png"
              + (f", {stem}_borders.png"
                 if args.predict and not args.ai else ""))


def run_ai(img, args):
    """Mask via the rembg neural model; returns (alpha, dist, tolerance)."""
    raw = ai_alpha(img)
    mask = (raw >= 0.5).astype(np.uint8) * 255
    mask = clean_mask(mask, 0, 0, args.min_area,
                      fill_holes=not args.keep_holes,
                      keep_largest=args.keep_largest)
    if mask.max() == 0:
        sys.exit("error: the AI model found no object (or it was smaller "
                 "than --min-area)")
    alpha = raw.copy()
    near = cv2.dilate(mask, np.ones((11, 11), np.uint8))
    alpha[near == 0] = 0.0          # drop discarded components' mattes
    if not args.keep_holes:
        alpha[(mask > 0) & (raw < 0.5)] = 1.0   # holes clean_mask filled
    if args.feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather)
    return np.clip(alpha, 0.0, 1.0), None, None


def run_color(img, args, stem):
    """The classic color-distance pipeline; returns (alpha, mask, dist,
    tolerance)."""
    lab = to_lab(img)
    if args.bg_color:
        bg_colors = []
        for spec in args.bg_color.split(";"):
            r, g_, b = (int(v) for v in spec.split(","))
            bg_colors.append(to_lab(np.array([[[b, g_, r]]], np.uint8))[0, 0])
        bg_colors = np.array(bg_colors, np.float32)
        samples = bg_samples(lab, args.bg_from, args.border_frac)
    else:
        samples = bg_samples(lab, args.bg_from, args.border_frac)
        bg_colors = estimate_bg_colors(samples, max_colors=args.bg_colors)
    shown = []
    for c in bg_colors:
        bgr = cv2.cvtColor(c.reshape(1, 1, 3).astype(np.uint8),
                           cv2.COLOR_LAB2BGR)[0, 0]
        shown.append(f"RGB({bgr[2]}, {bgr[1]}, {bgr[0]})")
    print("background color ≈ " + ", ".join(shown))

    if str(args.tolerance).lower() == "auto":
        tolerance = auto_tolerance(samples, bg_colors,
                                   args.lightness_weight, args.sat_weight)
        print(f"auto tolerance = {tolerance:.1f}")
    else:
        tolerance = float(args.tolerance)

    dist = color_distance(lab, bg_colors, args.lightness_weight,
                          args.sat_weight)
    mask = build_mask(dist, tolerance)
    mask = clean_mask(mask, args.open_size, args.close_size,
                      args.min_area, fill_holes=not args.keep_holes,
                      keep_largest=args.keep_largest)
    if mask.max() == 0:
        sys.exit("error: nothing left after thresholding — try lowering "
                 "--tolerance or checking --bg-color")

    if args.grabcut > 0:
        mask = grabcut_refine(img, mask, iters=args.grabcut)
        mask = clean_mask(mask, 0, 0, args.min_area,
                          fill_holes=not args.keep_holes,
                          keep_largest=args.keep_largest)
        if mask.max() == 0:
            sys.exit("error: GrabCut removed everything — try more "
                     "tolerance headroom or skip --grabcut")

    if args.predict:
        dbg = img.copy() if args.debug else None
        mask, bridged = predict_borders(
            mask, dist, tolerance, conf_ratio=args.conf_ratio,
            max_gap_frac=args.max_gap_frac, min_gap_px=args.min_gap,
            anchor_pts=args.anchor_pts, sample_depth=args.sample_depth,
            debug_img=dbg)
        print(f"border prediction: {bridged} indistinct gap(s) bridged")
        if dbg is not None:
            cv2.imwrite(f"{stem}_borders.png", dbg)

    alpha = make_alpha(mask, dist, tolerance, soft=args.soft,
                       soft_band=args.soft_band, feather=args.feather)
    return alpha, mask, dist, tolerance


if __name__ == "__main__":
    main()
