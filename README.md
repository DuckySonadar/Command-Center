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
rest of the app it needs two packages:

```bash
pip install opencv-python numpy
```

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
  object), higher keeps less. Default 28.
- `--bg-color R,G,B` — skip auto-detection when you know the backdrop.
- `--lightness-weight 0.5` — tolerate shadows/uneven lighting by caring
  less about brightness and more about hue.
- `--soft N` / `--feather N` — soft color ramp at the edge / Gaussian
  edge feathering instead of a hard cut.
- `--min-area`, `--open`, `--close`, `--keep-holes` — mask cleanup.
- `--debug` — dumps the mask, the color-distance map, and (with
  `--predict`) an overlay showing what the border predictor did.

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
