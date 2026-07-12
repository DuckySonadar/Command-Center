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
- `--soft N` / `--feather N` — soft color ramp at the edge / Gaussian
  edge feathering instead of a hard cut.
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
`--roi`, `--min-area`, `--keep-largest`, `--keep-holes` and `--feather`
still apply with `--ai`; the color options don't.

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

## Adding a new utility later

1. Write a function in `console.py` that does the work and returns a
   status string.
2. Add an `api_*` wrapper and register it in the `DISPATCH` dict.
3. Add a `<button data-endpoint="/api/your-task">` in the `PAGE` HTML.
