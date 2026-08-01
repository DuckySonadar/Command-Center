# Josiah's Maker Cave — Command Center

A small, dependency-free control panel for maintaining the website repo.
It runs a local web UI in your browser with buttons for common tasks.

## Run it

**Easiest:** double-click `Command Center.command` in Finder.
(First time only: right-click → Open, to get past the macOS "unidentified
developer" warning.)

**Or from a terminal:**

```bash
cd "Documents/Code/command-center"
python3 console.py
```

It opens http://127.0.0.1:8770 in your browser. Press `Ctrl-C` in the
terminal to stop it.

## Buttons

- **Build Manifest** — scans `inventory/` and regenerates
  `inventory-manifest.json` from the image filenames
  (`CODE-NNNN-Description-Designer.png`). Large previews, the
  `originals/` folder, and any non-matching files are skipped.
- **Commit & Push** — for **both** repos (the website and this app, since
  it's a living app): `git add -A`, commit (your message or an
  auto-generated one), `git pull --rebase`, then `git push`. A repo with
  no changes is skipped; a repo with no remote is committed but not pushed.
  Output for each repo is shown in the console panel.
- **Refresh git status** — shows `git status` for both repos.

## Repos

- **Website**: `mywebsiterepository-Iknowtotallyoriginal` (sibling folder)
- **Command Center**: this repo — https://github.com/DuckySonadar/Command-Center

If you move things, point the app at the website with an env var:
`MAKERCAVE_REPO=/path/to/repo python3 console.py`

Push uses your normal git credentials (the macOS Keychain helper). If a
push ever fails for auth, run one `git push` manually in the terminal to
refresh the stored token.

## Background removal (`bg_remove.py`)

A standalone command-line tool that cuts objects out of a photo with a
predominantly uniform background and writes a transparent PNG. Unlike the
rest of the app it needs two packages (plus two optional ones for
iPhone HEIC/HEIF photos):

```bash
pip install opencv-python numpy
pip install pillow pillow-heif   # optional: HEIC/HEIF input support
```

Input can be anything OpenCV reads (JPEG, PNG, TIFF, ...) and, with the
optional packages, HEIC/HEIF straight off an iPhone. Output is always
PNG, since it needs an alpha channel.

Quick start (generates a synthetic test image so you can try it
immediately):

```bash
python3 bg_remove.py --demo demo.png
python3 bg_remove.py demo.png cutout.png --predict --debug
python3 bg_remove.py photo.jpg                 # writes photo_cutout.png
```

How it works: the background color is estimated from the image border
(k-means on a strip around the edge), every pixel is scored by its
perceptual (CIE Lab) distance from that color, and pixels beyond
`--tolerance` become the object mask. The mask is then cleaned with
morphology, small-blob removal and hole filling.

Most useful knobs (see `--help` for all of them):

- `--tolerance N` — the big one. Lower keeps more (soft shadows become
  object), higher keeps less. Default 28. `--tolerance auto` derives a
  starting value from the border pixels themselves; treat it as a first
  guess and nudge from there.
- `--bg-colors N` — model the background as up to N colors instead of 1.
  Use 2–4 when the backdrop has a gradient, vignette, or uneven
  lighting, so the whole *range* of background shades is matched.
- `--bg-color "R,G,B"` — skip auto-detection when you know the backdrop;
  give several separated by `;` to specify the range manually.
- `--lightness-weight` / `--sat-weight` — reweight what "different from
  the background" means. Lower `--lightness-weight` (e.g. 0.5) to
  ignore shadows and uneven lighting; raise `--sat-weight` (e.g. 2)
  when the object differs from the backdrop mainly in saturation/hue
  rather than brightness. The two combine well for subtle objects.
- `--bg-from WHERE` — where the background is sampled: `border`
  (default), `image` (dominant color of the whole frame), or a point
  like `0.5,0.9` (fractional x,y of a clean patch of backdrop). Use the
  latter two when things other than backdrop touch the frame edges.
- `--roi FX,FY,FW,FH` — only look for objects inside this fractional
  rectangle; everything outside becomes transparent. The practical fix
  for desk clutter, or reflections in the corners of a glossy table.
- `--keep-largest N` — keep only the N biggest objects found.
- `--grabcut 3` — refine the color mask with OpenCV's GrabCut (color
  mixture models + spatial smoothness). Great at snapping off attached
  shadows and reflections that share a tint with the object.
- `--shrink N` — pull the cut line inward by N pixels (a compositor's
  "choke"). *This*, not soft/feather, is the fix when a rim of leftover
  background hugs the object. Start with 5–15 on 12 MP photos; negative
  values expand instead.
- `--soft N` / `--feather N` — soft color ramp at the edge / Gaussian
  edge feathering instead of a hard cut. These change how *gradual* the
  existing cut line is; they don't move it. `--soft` is in Lab color
  units (5–15 is sensible); `--feather` is a blur radius in pixels
  (2–4 is subtle on a 12 MP photo, 8+ clearly soft).
- `--min-area`, `--open`, `--close`, `--keep-holes` — mask cleanup.
- `--debug` — dumps the mask, the color-distance map, a preview of the
  cutout over a checkerboard, and (with `--predict`) an overlay showing
  what the border predictor did.

### AI engine (`--ai`)

For photos that defeat color logic entirely — e.g. a blue-gray object on
a mat whose glare reflects the same blue — there's a neural option:

```bash
pip install onnxruntime   # once; first --ai run downloads ~170 MB
python3 bg_remove.py IMG_2140.HEIC out.png --ai --keep-largest 1
```

It runs fully locally (U^2-Net). Plain `onnxruntime` is all it needs —
the tool ships its own tiny model runner. If the `rembg` package happens
to be installed it is used instead, but it's *not* required (rembg pulls
in numba/llvmlite, which don't have prebuilt wheels on every
Python/macOS combination and then demand cmake + LLVM to compile).
`--roi`, `--min-area`, `--keep-largest`, `--keep-holes`, `--shrink` and
`--feather` still apply with `--ai`; the color options don't.

U^2-Net works on a 320x320 grid internally, so on a 12 MP photo its edge
is naturally a wide soft band. To tighten a halo: raise
`--ai-threshold` (0.5 default → try 0.7), add `--shrink 8-15`, and
finish with a small `--feather 2-3`.

### Recipes from the test photos

Worked starting points, tuned on the photos in this repo:

```bash
# single toy on a dark table, attached shadow (IMG_2128)
python3 bg_remove.py IMG_2128.HEIC out.png --bg-colors 3 --tolerance auto \
  --lightness-weight 0.4 --sat-weight 2 --min-area 5000 --keep-largest 1 \
  --grabcut 3 --feather 2

# several colorful toys on a glossy black table, reflections in the
# corners (IMG_2154): fence the toys in with --roi
python3 bg_remove.py IMG_2154.jpg out.png --bg-colors 3 --tolerance 20 \
  --lightness-weight 0.5 --roi 0.13,0.1,0.65,0.78 --min-area 5000 \
  --keep-largest 5 --feather 2

# same table, toy reflections attached below (IMG_2155): GrabCut snaps
# the reflections off
python3 bg_remove.py IMG_2155.jpg out.png --bg-colors 3 --tolerance 26 \
  --lightness-weight 0.6 --sat-weight 1.2 --roi 0.05,0.12,0.92,0.72 \
  --min-area 5000 --keep-largest 4 --grabcut 3 --feather 2

# glow-in-the-dark toys, dim room (IMG_2161): bright objects, dark bg,
# so tolerance can be generous
python3 bg_remove.py IMG_2161.jpeg out.png --bg-colors 3 --tolerance 40 \
  --min-area 5000 --keep-largest 2 --feather 2

# toy on a leather mat whose glare reflects the toy's own blue
# (IMG_2140, IMG_2142): color logic can't separate — use the AI engine
python3 bg_remove.py IMG_2140.HEIC out.png --ai --keep-largest 1 --feather 2
```

General tuning order: get the background model right first (`--bg-from`,
`--bg-colors`), then set `--tolerance` (start with `auto`, check the
`--debug` preview), then clean up (`--roi`, `--keep-largest`,
`--min-area`), and reach for `--grabcut` when shadows or reflections
stay attached to the object.

### Border prediction (`--predict`)

Experimental. Where an object fades into the background (a white edge on
a white backdrop, a soft shadowed side), simple thresholding eats a bite
out of the silhouette. With `--predict` the tool walks each object's
outline, measures how *distinct* each border point is (color contrast
across the edge), finds contiguous indistinct stretches, and bridges each
one with a cubic Hermite curve anchored on the distinct border either
side — using those segments' tangent directions to approximate what the
hidden border most plausibly looks like. Tune with `--conf-ratio`
(what counts as indistinct), `--max-gap-frac` (longest gap it will dare
to invent), and `--anchor-pts` (how much of the good border defines each
tangent). The `--debug` overlay draws distinct border green, indistinct
red, and predicted curves magenta.

## Flexi fish generator (`flexifish.py`, `flexifish_nurbs.py`)

Two standalone tools that generate print-in-place articulated fish as
binary STLs (no supports, flat belly on the plate), plus two browser
designers (see the next section). The Python tools need:

```bash
pip install numpy scikit-image
pip install matplotlib        # optional, only for --png renders
```

`flexifish.py` is the original: the body is sculpted from smooth-blended
blobs and everything is a numeric parameter ("slider") — see
`--list-params`. `flexifish_nurbs.py` keeps all of its joint machinery
and sliders but replaces the sculpting: **you draw the fish with NURBS
curves in two views**, and named regions are derived from the drawing.

```bash
python3 flexifish_nurbs.py                          # default fish
python3 flexifish_nurbs.py --preview --png --svg    # fast look
python3 flexifish_nurbs.py --dump-shape             # editable template
python3 flexifish_nurbs.py --shape my_fish.json --coupon
```

### The curves (what you draw)

All coordinates are millimeters. Side view is (x, z) with z = 0 the
build plate; top view is (x, y) with y the half-width (right side only —
it's mirrored).

| curve          | view | kind   | what it is                              |
|----------------|------|--------|-----------------------------------------|
| `back`         | side | open   | nose tip → caudal root, top silhouette  |
| `belly`        | side | open   | nose tip → caudal root; dip below z=0, the plate cut makes the flat belly |
| `plan`         | top  | open   | nose tip → caudal root, half-width      |
| `dorsal_fin`   | side | closed | drawn overlapping the back so it fuses  |
| `caudal_fin`   | side | closed | the tail fan — draw the fork right in   |
| `pectoral_fin` | top  | closed | front paddle, drawn in place, overlapping the body edge |
| `pelvic_fin`   | top  | closed | rear paddle, same rules                 |

Each curve is `{"points": [[x,y],...], "degree": 3, "weights": [...]}`
(degree and weights optional; weights > 1 pull the curve toward a
point). The shape JSON is deep-merged over the built-in default, so a
file containing only a new `dorsal_fin` changes just that; set a fin
to `null` to delete it. `--svg` writes both views with the sampled
curves, control cages, region bands and joint cuts — the fastest way
to see what you drew.

### The regions (derived, not typed in)

Everything ahead of the dorsal fin outline is the **head** region: it
stays rigid and carries the eyes and mouth. The lower head region holds
the **pectoral** and **pelvic** fin regions — both paddles must attach
there (validated), since their ball sockets can't straddle a joint cut.
The dorsal outline claims the **dorsal** region: exactly one articulated
segment with the fin centered in it. The **tail** region is the only one
with a variable segment count (`regions.tail_segments`), and the
**caudal** region is the solid tail-root + fan piece. Draw the dorsal
fin far forward and its region simply fuses into the rigid head, with
the whole span behind it becoming tail segments.

### The sliders (kept from the original)

Per-region scales that transform the drawn curves, smoothly blended at
region borders — tweak inline with `--set`:

```bash
python3 flexifish_nurbs.py --set tail.length=1.3 --set head.height=0.9 \
    --set dorsal.fin_height=1.5 --set regions.tail_segments=5 \
    --set head.mouth_open=0.6 --set caudal.thickness=1.2
```

`head/dorsal/tail` have `.length .width .height`; `dorsal` adds
`.fin_height`; `caudal` has `.length .height .thickness`;
`pectoral/pelvic` have `.length .width` (about their attachment).
Joint clearances, eyes, walls etc. are still `FishParams`
(`--config`, `--list-params`, same as flexifish.py).

The caudal fin's `length`/`height` sliders act through a logistic
falloff — zero effect where the fin meets the tail root (the fused zone
never distorts), full effect at the farthest edge.

The mouth has three shapes (the `"mouth"` section of the shape JSON):
`"curve"` (default) carves a groove along a NURBS `mouth` curve you
draw in the side view, swept across the nose; `"groove"` is the same
cut on a plain plane instead of a drawn curve; `"pucker"` is the
original torus lips. Both groove modes share three controls —
`mouth.height` (mm up/down), `mouth.tilt` (degrees from vertical,
rotating about the arc midpoint; past 45° you get an overhang/print
warning) and `mouth.length` (mm) — and the cut always stops 2 mm above
the build plate. `head.mouth_open` still carves an open-mouth pocket on
top of any of them.

### Segment joints

Segments are linked by a **plate-cut ball and socket** — the same
principle as the side fins. A ball on a short neck belongs to the front
segment; the rear segment carries a spherical shell around it. There is
no vertical post: everything is concentric on the ball centre, so there
is no lever arm to pry a joint apart, and the shell is a sphere that
gets clipped to the body silhouette (0.15 mm inside it) and therefore
can never bulge out through the skin.

Capture comes from the socket mouth being deliberately *narrower* than
the ball. The mouth is a cone opened only as wide as the requested
swing needs, so grip is usually well above the floor:

| knob | default | what it does |
|------|---------|--------------|
| `joint_capture` | 0.20 | minimum grip as a fraction of ball radius (also never below 0.45 mm) |
| `joint_ball_max` | 5.0 | largest ball radius, mm |
| `joint_neck` | 0.40 | neck radius ÷ ball radius — thinner neck buys more swing |
| `clearance` | 0.55 | printed gap (PETG ~0.55, PLA ~0.45) |

Raise `joint_capture` for a stiffer link, lower `swing_front_deg` /
`swing_rear_deg` to spend the same geometry on grip instead of travel.
On the default fish that yields 0.56–1.46 mm of grip (22–31% of the
ball radius) while still hitting the full requested 12–16° of swing.

The ball rests on the build plate so it prints unsupported, and the
generator guarantees **more than 80% of each ball sits above the plate**
(typically 87–98%). Ball size is chosen by bisection against every
constraint at once: it must fit inside the body at its own height, under
the back, and end-to-end with the neighbouring joint's shell — that last
one is usually what binds, so *longer segments*, not a wider body, are
what buy bigger joints.

Print notes: run `--coupon` first and print the one-joint test;
`--preview` is for looking only (at 0.62 mm voxels the 0.55 mm joint
clearances fuse — the STL will report fewer shells than expected).
A full-res build reports `shells=` and `manifold=`; the expected shell
count is printed next to it, and a mismatch means fused or orphaned
parts — inspect before printing.

## Fish designers in the browser (`fish_designer.html`, `fish_designer_nurbs.html`)

Two self-contained web apps (no server, no dependencies — just open the
file in a browser). Both show a live 3D preview with segment-cut
grooves (display only — the grooves and welded side fins never touch
the printable geometry) and run the same layout/joint validation as
the Python tools.

- **`fish_designer.html`** drives the original blob fish: sliders only,
  exports a `--config` JSON for `flexifish.py`.
- **`fish_designer_nurbs.html`** drives the NURBS fish and adds the
  drawing interface: **Draw side** and **Draw top** modes show the
  curves with draggable control points (tap a point, drag it; ＋/−
  point buttons edit the active curve; × on a fin chip removes that
  fin). Region bands (head / dorsal / tail / caudal) and joint cut
  lines update live as you draw, the region sliders sit in the right
  panel (including the mouth-shape dropdown), and **Shape JSON**
  exports a file for `flexifish_nurbs.py --shape`. **Save STL** builds
  the real printable plate in the browser — segmented body, joint pins
  and cavities, ball-socket fin parts — at the same 0.3 mm resolution
  the Python tool prints at (expect a few seconds; its surface-nets
  mesher can leave a handful of non-manifold edges that slicers repair
  automatically, so the Python build stays the pristine path). The JS
  is a numerically faithful port of the Python pipeline (same curves,
  same regions, same joint sizing, same plate), so what you draw is
  what prints.

On a phone the panel docks to the bottom half; one finger drags points,
two fingers pan/zoom the drawing.

The NURBS designer is also published on the website's **SDF EDITOR**
section, as `tools/fish-editor-nurbs.html` in the
`mywebsiterepository-Iknowtotallyoriginal` repo, alongside the general
`sdf_editor.html` modeller. The blob designer is not published. Those are
plain copies — this repo stays the source of truth, so edit here and
re-copy after a change.

## Adding a new utility later

1. Write a function in `console.py` that does the work and returns a
   status string.
2. Add an `api_*` wrapper and register it in the `DISPATCH` dict.
3. Add a `<button data-endpoint="/api/your-task">` in the `PAGE` HTML.
