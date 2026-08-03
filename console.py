#!/usr/bin/env python3
"""
Josiah's Maker Cave — Command Center
====================================
A tiny, dependency-free control panel for managing the website repo.

It serves a local web UI (http://127.0.0.1:8770) with buttons that run
maintenance tasks against the website repository:

  1. BUILD MANIFEST  — regenerate inventory-manifest.json from the images
                       in the inventory/ folder.
  2. COMMIT & PUSH   — git add -A, commit, pull --rebase, and push.
  3. PUBLISH TOOLS   — copy the editors into the website repo.

It also serves the editors themselves (MetaMeld and the fish designers) at
/tools/<slug>, so they open in a click instead of being hunted for on disk.

Adding a new utility later is three small steps (see ADD-A-UTILITY below).

Run it:  python3 console.py   (or double-click the .command launcher)
Stop it: Ctrl-C in the terminal.

By default it listens on 127.0.0.1, so nothing here is reachable from the
network. MAKERCAVE_HOST=0.0.0.0 opens it up so a phone on the same Wi-Fi can
load the editors — see the note by that setting before you use it.
"""

import os
import re
import json
import socket
import subprocess
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── Configuration ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The website repo is, by default, a sibling folder of this app.
DEFAULT_REPO = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "mywebsiterepository-Iknowtotallyoriginal")
)
WEBSITE_REPO = os.environ.get("MAKERCAVE_REPO", DEFAULT_REPO)
# This app ("Command Center") is a "living app" — its own folder is a git
# repo too, and gets committed & pushed alongside the website.
COMMAND_CENTER_REPO = SCRIPT_DIR
# Repos the Commit & Push button operates on, in order.
REPOS = [("Website", WEBSITE_REPO), ("Command Center", COMMAND_CENTER_REPO)]

INVENTORY = os.path.join(WEBSITE_REPO, "inventory")
MANIFEST = os.path.join(WEBSITE_REPO, "inventory-manifest.json")
PORT = int(os.environ.get("MAKERCAVE_PORT", "8770"))
# Loopback by default. Setting this to 0.0.0.0 lets a phone on the same Wi-Fi
# open the editors without a second web server -- but it exposes Commit &
# Push and Build Manifest to everything else on that network too, so only do
# it on a network you trust, and stop the app when you are done.
HOST = os.environ.get("MAKERCAVE_HOST", "127.0.0.1")

# ── The editors ───────────────────────────────────────────────
# Single-file apps that live in this repo: no build step, no dependencies,
# nothing path-relative. Command Center does two things with them -- serves
# them (so they are one click away, and reachable from a phone when HOST
# allows it), and copies them into the website repo.
#
# `publish` is the path inside the website repo, or None for a tool that is
# deliberately not on the site. That copy used to live only in the README as
# "re-copy after a change", which is exactly the kind of step that gets
# skipped; here it is a button that also tells you when it was already
# current.
TOOLS = [
    {"slug": "metameld", "file": "sdf_editor.html", "name": "MetaMeld",
     "blurb": "SDF modeller for phones — build a shape from primitives, "
              "raymarch it live, mesh it to a printable STL.",
     "publish": "sdf_editor.html"},
    {"slug": "fish-nurbs", "file": "fish_designer_nurbs.html",
     "name": "Fish Designer · NURBS",
     "blurb": "Draw the flexi fish in side and top views; regions, joints "
              "and the printable plate follow the curves.",
     "publish": "tools/fish-editor-nurbs.html"},
    {"slug": "fish-blob", "file": "fish_designer.html",
     "name": "Fish Designer · blob",
     "blurb": "The slider-driven blob fish. Not published to the site — "
              "it is the simpler sibling, kept for quick shape trials.",
     "publish": None},
]
TOOLS_BY_SLUG = {t["slug"]: t for t in TOOLS}

# Designer-name spellings to normalize for display (the file on disk is
# left untouched; only the manifest's "designer" field is corrected).
DESIGNER_FIXES = {"cinderwing3d": "Cinderwing3D", "flexseeds": "FlexSeeds"}

ITEM_RE = re.compile(r"^([A-Z]{3})-(\d{4})-([^-]+)-([^-]+)\.png$", re.IGNORECASE)


# ── Core tasks ────────────────────────────────────────────────
def build_manifest():
    """Regenerate inventory-manifest.json from inventory/*.png filenames."""
    entries, skipped = [], []
    for name in sorted(os.listdir(INVENTORY)):
        if name.startswith(".") or name.startswith("Large-"):
            continue
        if os.path.isdir(os.path.join(INVENTORY, name)):
            continue  # e.g. the gitignored originals/ folder
        m = ITEM_RE.match(name)
        if not m:
            skipped.append(name)
            continue
        code, num, desc, designer = m.groups()
        entries.append({
            "category": code.upper(),
            "number": num,
            "description": desc,
            "designer": DESIGNER_FIXES.get(designer.lower(), designer),
            "file": name,
        })
    with open(MANIFEST, "w") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")
    return entries, skipped


def run_git(args, cwd):
    """Run a git command in `cwd`; return (returncode, combined output)."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def has_remote(cwd):
    rc, out = run_git(["remote"], cwd)
    return rc == 0 and bool(out.strip())


def default_commit_message():
    return f"Update via Control Console — {datetime.now():%Y-%m-%d %H:%M}"


def commit_and_push_one(label, cwd, message):
    """Stage, commit (if needed), rebase, and push a single repo."""
    lines = [f"━━━ {label} ━━━  {cwd}"]
    if not os.path.isdir(os.path.join(cwd, ".git")):
        lines.append("Not a git repo — skipped.\n")
        return "\n".join(lines)

    def step(label_, args):
        rc, out = run_git(args, cwd)
        lines.append(f"$ git {label_}\n{out or '(no output)'}\n")
        return rc

    step("add -A", ["add", "-A"])

    _, status = run_git(["status", "--porcelain"], cwd)
    if status.strip():
        if step("commit", ["commit", "-m", message]) != 0:
            lines.append("⚠️  Commit failed — skipping push for this repo.")
            return "\n".join(lines)
    else:
        lines.append("Working tree clean — nothing new to commit.\n")

    if not has_remote(cwd):
        lines.append("No git remote configured — skipping push.")
        return "\n".join(lines)

    step("pull --rebase", ["pull", "--rebase"])
    rc = step("push", ["push"])
    lines.append("✅ Pushed." if rc == 0 else "⚠️  Push failed — see output above.")
    return "\n".join(lines)


def commit_and_push(message):
    """Commit & push every repo in REPOS (website + this living app)."""
    return "\n\n".join(
        commit_and_push_one(label, cwd, message) for label, cwd in REPOS
    )


def publish_tools():
    """Copy each publishable editor into the website repo.

    Reports every tool, not just the ones it touched, because the question
    worth answering is usually "is the site stale?" rather than "copy it" --
    and a run that changes nothing should say so plainly rather than look
    like it did something.
    """
    if not os.path.isdir(WEBSITE_REPO):
        return (f"⚠️  Website repo not found at:\n    {WEBSITE_REPO}\n"
                f"Set MAKERCAVE_REPO to its path and retry.")
    lines, changed = [], 0
    for t in TOOLS:
        if not t["publish"]:
            lines.append(f"·  {t['name']} — not published, by design")
            continue
        src = os.path.join(SCRIPT_DIR, t["file"])
        dst = os.path.join(WEBSITE_REPO, t["publish"])
        if not os.path.isfile(src):
            lines.append(f"⚠️  {t['name']} — {t['file']} is missing from "
                         f"this repo")
            continue
        with open(src, "rb") as fh:
            data = fh.read()
        old = None
        if os.path.isfile(dst):
            with open(dst, "rb") as fh:
                old = fh.read()
        if old == data:
            lines.append(f"=  {t['name']} — already current "
                         f"({t['publish']}, {len(data):,} bytes)")
            continue
        os.makedirs(os.path.dirname(dst) or WEBSITE_REPO, exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        lines.append(f"✅ {t['name']} — {'updated' if old else 'created'} "
                     f"{t['publish']} ({len(data):,} bytes)")
        changed += 1
    lines.append("")
    lines.append(f"{changed} file(s) copied. Copying only writes them into "
                 f"the website\nfolder — run Commit & Push to put them on "
                 f"the site." if changed else
                 "Nothing to copy — the website already has this build.")
    return "\n".join(lines)


# ── ADD-A-UTILITY ─────────────────────────────────────────────
# To add a new button later:
#   1. Write a function above that returns a status string.
#   2. Add an entry to API below: "/api/your-task": your_function
#   3. Add a <button data-endpoint="/api/your-task"> in PAGE.
API = {
    "/api/build-manifest": "build_manifest",
    "/api/commit-push": "commit_push",
    "/api/publish-tools": "publish_tools",
    "/api/status": "status",
}


def api_build_manifest(_payload):
    entries, skipped = build_manifest()
    cats = {}
    for e in entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    out = [f"Wrote {len(entries)} item(s) to inventory-manifest.json.",
           "By category: " + ", ".join(f"{k} {v}" for k, v in sorted(cats.items()))]
    if skipped:
        out.append("\nSkipped (not inventory items):\n  " + "\n  ".join(skipped))
    return {"ok": True, "output": "\n".join(out)}


def api_commit_push(payload):
    message = (payload.get("message") or "").strip() or default_commit_message()
    return {"ok": True, "output": commit_and_push(message)}


def api_publish_tools(_payload):
    return {"ok": True, "output": publish_tools()}


def api_status(_payload):
    blocks = []
    for label, cwd in REPOS:
        rc, out = run_git(["status", "--short", "--branch"], cwd)
        blocks.append(f"━━━ {label} ━━━\n{out or 'clean'}")
    return {"ok": True, "output": "\n\n".join(blocks)}


DISPATCH = {
    "/api/build-manifest": api_build_manifest,
    "/api/commit-push": api_commit_push,
    "/api/publish-tools": api_publish_tools,
    "/api/status": api_status,
}


# ── HTTP server ───────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_tool(self, slug):
        """Serve one editor. Read from disk on every request and marked
        no-store: editing the file and hitting reload is the whole workflow,
        and a cached copy would quietly defeat it."""
        tool = TOOLS_BY_SLUG.get(slug)
        if not tool:
            self._send(404, "<h1>No such tool</h1>", "text/html; charset=utf-8")
            return
        try:
            with open(os.path.join(SCRIPT_DIR, tool["file"]), "rb") as fh:
                body = fh.read()
        except OSError as exc:
            self._send(404, f"<h1>{tool['name']} is missing</h1><pre>{exc}</pre>",
                       "text/html; charset=utf-8")
            return
        self._send(200, body, "text/html; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, render_page(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps(api_status({})))
        elif path.startswith("/tools/"):
            self._send_tool(path[len("/tools/"):])
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")   # browsers always ask
        else:
            self._send(404, json.dumps({"ok": False, "output": "Not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode() if length else ""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        handler = DISPATCH.get(self.path)
        if not handler:
            self._send(404, json.dumps({"ok": False, "output": "Not found"}))
            return
        try:
            self._send(200, json.dumps(handler(payload)))
        except Exception as exc:  # surface errors in the UI instead of crashing
            self._send(200, json.dumps({"ok": False, "output": f"Error: {exc}"}))

    def log_message(self, *_):  # keep the terminal quiet
        pass


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Maker Cave Command Center</title>
<style>
  :root {
    --orange:#F54D27; --yellow:#F5E427; --mint:#25E8BB; --purple:#B038D1;
    --display:'Impact','Haettenschweiler','Arial Narrow Bold',sans-serif;
    --mono:ui-monospace,'Courier New',monospace;
    --body:'Inter',system-ui,'Helvetica Neue',Arial,sans-serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#000; color:#e9e9e9; font-family:var(--body);
         padding:40px 48px 60px; min-height:100vh; }
  .kicker { font-family:var(--mono); font-size:.72rem; letter-spacing:.22em;
            color:#8a8a8a; text-transform:uppercase; }
  h1 { font-family:var(--display); font-size:clamp(2.4rem,6vw,4rem);
       text-transform:uppercase; letter-spacing:.02em; color:#fff;
       line-height:.92; margin:6px 0 36px; }
  h1 .a { color:var(--orange); } h1 .b { color:var(--mint); }
  .actions { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
             gap:28px; max-width:900px; }
  .card { display:flex; flex-direction:column; gap:14px; }
  .card h2 { font-family:var(--display); font-size:1.5rem; text-transform:uppercase;
             letter-spacing:.02em; }
  .card p { font-size:.95rem; line-height:1.5; color:#bdbdbd; }
  .card.manifest h2 { color:var(--mint); }
  .card.push h2 { color:var(--orange); }
  .card.publish h2 { color:var(--purple); }
  h3.section { font-family:var(--display); font-size:1.5rem; text-transform:uppercase;
               letter-spacing:.02em; color:var(--yellow); margin:44px 0 6px; }
  h3.section + p { font-size:.95rem; line-height:1.5; color:#bdbdbd;
                   max-width:900px; margin-bottom:22px; }
  .tools { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
           gap:28px; max-width:900px; }
  .tool { display:flex; flex-direction:column; gap:10px; border-left:4px solid #2a2a2a;
          padding-left:16px; }
  .tool h4 { font-family:var(--display); font-size:1.15rem; text-transform:uppercase;
             letter-spacing:.02em; color:#fff; font-weight:normal; }
  .tool p { font-size:.9rem; line-height:1.5; color:#bdbdbd; }
  .tool .where { font-family:var(--mono); font-size:.72rem; letter-spacing:.06em;
                 color:#8a8a8a; }
  .tool .missing { color:var(--orange); }
  .tool a { font-family:var(--display); font-size:1.05rem; text-transform:uppercase;
            letter-spacing:.03em; color:#000; background:var(--yellow);
            text-decoration:none; padding:9px 18px; align-self:start;
            transition:transform .1s; }
  .tool a:hover { transform:translateY(-2px); }
  .publish button { background:var(--purple); color:#fff; }
  input[type=text] { font-family:var(--mono); font-size:.85rem; background:#0d0d0d;
                     color:#fff; border:2px solid #2a2a2a; padding:10px 12px; }
  input[type=text]:focus { outline:none; border-color:var(--yellow); }
  button { font-family:var(--display); font-size:1.3rem; text-transform:uppercase;
           letter-spacing:.03em; color:#000; border:none; padding:14px 24px;
           cursor:pointer; align-self:start; transition:transform .1s, filter .1s; }
  button:hover:not(:disabled) { transform:translateY(-2px); }
  button:disabled { opacity:.5; cursor:wait; }
  .manifest button { background:var(--mint); }
  .push button { background:var(--orange); }
  .console-wrap { margin-top:44px; max-width:900px; }
  .console-head { display:flex; justify-content:space-between; align-items:baseline;
                  margin-bottom:10px; }
  .console-head span { font-family:var(--mono); font-size:.7rem; letter-spacing:.18em;
                       color:#8a8a8a; text-transform:uppercase; }
  #refresh { background:none; color:var(--yellow); font-family:var(--mono);
             font-size:.7rem; letter-spacing:.15em; padding:0; }
  #refresh:hover:not(:disabled) { transform:none; text-decoration:underline; }
  pre#out { font-family:var(--mono); font-size:.82rem; line-height:1.5; color:#cfcfcf;
            background:#0a0a0a; border-left:4px solid var(--purple); padding:18px 20px;
            white-space:pre-wrap; word-break:break-word; min-height:120px; }
</style>
</head>
<body>
  <p class="kicker">JOSIAH'S MAKER CAVE</p>
  <h1>COMMAND <span class="a">CEN</span><span class="b">TER</span></h1>

  <div class="actions">
    <div class="card manifest">
      <h2>Build Manifest</h2>
      <p>Scan the inventory folder and regenerate inventory-manifest.json from the image filenames.</p>
      <button data-endpoint="/api/build-manifest">Build Manifest</button>
    </div>

    <div class="card publish">
      <h2>Publish Tools</h2>
      <p>Copy the editors into the website repo. Says which were already current;
         Commit &amp; Push is what actually puts them on the site.</p>
      <button data-endpoint="/api/publish-tools">Publish Tools</button>
    </div>

    <div class="card push">
      <h2>Commit &amp; Push</h2>
      <p>Stage, commit, rebase, and push both repos — the website and this app — to GitHub.</p>
      <input type="text" id="commit-msg" placeholder="Commit message (optional)" />
      <button data-endpoint="/api/commit-push" data-msg="commit-msg">Commit &amp; Push</button>
    </div>
  </div>

  <h3 class="section">Editors</h3>
  <p>Served straight from this repo, read fresh on every load — edit the file,
     reload the tab. To use one on a phone, restart with
     <code>MAKERCAVE_HOST=0.0.0.0</code> and open the address printed in the
     terminal; that also exposes the buttons above to the network, so only do
     it on Wi-Fi you trust.</p>
  <div class="tools"><!--TOOLS--></div>

  <div class="console-wrap">
    <div class="console-head">
      <span>Output</span>
      <button id="refresh">↻ Refresh git status</button>
    </div>
    <pre id="out">Ready.</pre>
  </div>

<script>
  const out = document.getElementById("out");
  function log(text) { out.textContent = text; }

  async function call(endpoint, body) {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    return res.json();
  }

  document.querySelectorAll("button[data-endpoint]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const all = document.querySelectorAll("button");
      all.forEach(b => b.disabled = true);
      log("Working…");
      try {
        const body = {};
        if (btn.dataset.msg) body.message = document.getElementById(btn.dataset.msg).value;
        const data = await call(btn.dataset.endpoint, body);
        log((data.ok ? "" : "⚠️ ") + data.output);
      } catch (e) {
        log("Error: " + e.message);
      } finally {
        all.forEach(b => b.disabled = false);
      }
    });
  });

  document.getElementById("refresh").addEventListener("click", async () => {
    log("Checking…");
    const res = await fetch("/api/status");
    const data = await res.json();
    log("git status:\\n" + data.output);
  });
</script>
</body>
</html>
"""


def render_page():
    """PAGE with the editor cards filled in.

    Built per request rather than once at import: it reports whether each file
    is actually on disk, and that is the sort of thing you want to be true
    now rather than true at startup.
    """
    cards = []
    for t in TOOLS:
        here = os.path.isfile(os.path.join(SCRIPT_DIR, t["file"]))
        where = (f"→ {t['publish']} on the site" if t["publish"]
                 else "not published to the site")
        cards.append(
            f'<div class="tool">'
            f'<h4>{t["name"]}</h4>'
            f'<p>{t["blurb"]}</p>'
            + (f'<a href="/tools/{t["slug"]}" target="_blank" '
               f'rel="noopener">Open</a>'
               if here else
               f'<p class="where missing">{t["file"]} is missing from this '
               f'folder</p>')
            + f'<p class="where">{t["file"]} &nbsp;·&nbsp; {where}</p>'
            f'</div>')
    return PAGE.replace("<!--TOOLS-->", "".join(cards))


def lan_address():
    """This machine's address on the local network, or None.

    No traffic is sent -- connecting a UDP socket only picks the route the
    kernel would use, which is what names the right interface on a machine
    with several."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))       # TEST-NET-1: reserved, unrouted
            return s.getsockname()[0]
    except OSError:
        return None


def main():
    if not os.path.isdir(WEBSITE_REPO):
        print(f"⚠️  Website repo not found at:\n    {WEBSITE_REPO}\n"
              f"Set MAKERCAVE_REPO to its path and retry.")
    url = f"http://127.0.0.1:{PORT}/"
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print(f"Could not start on {HOST}:{PORT}: {exc}\n"
              f"Another copy may already be running at {url}")
        return
    print("┌─────────────────────────────────────────────┐")
    print("│  Josiah's Maker Cave — Command Center        │")
    print("└─────────────────────────────────────────────┘")
    print(f"  Website        : {WEBSITE_REPO}")
    print(f"  Command Center : {COMMAND_CENTER_REPO}")
    print(f"  Open           : {url}")
    for t in TOOLS:
        print(f"    {t['name']:<24} {url}tools/{t['slug']}")
    if HOST not in ("127.0.0.1", "localhost"):
        ip = lan_address()
        print(f"\n  ⚠️  Listening on {HOST} — reachable from the network.")
        if ip:
            print(f"      On a phone, same Wi-Fi:  http://{ip}:{PORT}/")
        print("      Build Manifest and Commit & Push are reachable too.")
        print("      Stop the app when you are done.")
    print("  Stop           : press Ctrl-C\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down. Bye!")
        server.shutdown()


if __name__ == "__main__":
    main()
