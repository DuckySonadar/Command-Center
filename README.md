# Josiah's Maker Cave — Control Console

A small, dependency-free control panel for maintaining the website repo.
It runs a local web UI in your browser with buttons for common tasks.

## Run it

**Easiest:** double-click `Maker Cave Console.command` in Finder.
(First time only: right-click → Open, to get past the macOS "unidentified
developer" warning.)

**Or from a terminal:**

```bash
cd "Documents/Code/maker-cave-console"
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

## Git remote for this app

This folder is its own git repo. To enable pushing it, create an empty
repo on GitHub (no README/license), then point this repo at it:

```bash
cd "Documents/Code/maker-cave-console"
git remote add origin https://github.com/<you>/maker-cave-console.git
git push -u origin main
```

After that, the Commit & Push button keeps it in sync automatically.

## Notes

- The app expects the website repo to be a sibling folder named
  `mywebsiterepository-Iknowtotallyoriginal`. If you move things, point it
  at the repo with an env var:
  `MAKERCAVE_REPO=/path/to/repo python3 console.py`
- Push uses your normal git credentials (the macOS Keychain helper). If a
  push ever fails for auth, run one `git push` manually in the terminal to
  refresh the stored token.

## Adding a new utility later

1. Write a function in `console.py` that does the work and returns a
   status string.
2. Add an `api_*` wrapper and register it in the `DISPATCH` dict.
3. Add a `<button data-endpoint="/api/your-task">` in the `PAGE` HTML.
