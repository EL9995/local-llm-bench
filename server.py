#!/usr/bin/env python3
"""Local web page for the bench: type a task, pick models, and watch every model's answer arrive.
Runs blind, orchestrator, vision (read a photo, then hand the text on) and debate ("roundtable"
in code) modes.

Run:   python3 server.py        then open http://localhost:8765
"""

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import bench
import report

PORT = 8765
PAGE = Path(__file__).with_name("ui.html")
MODES = ("blind", "orchestrator", "vision", "roundtable")  # the page checks this, so a stale server says so
RAM_GB = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)  # GiB, as Apple advertises it
MAX_BODY = 60 * 2**20  # uploaded photos, base64-encoded


def caps(name):
    try:
        return bench.capabilities(name)
    except OSError:
        return []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif url.path == "/models":
            try:
                tags = bench.post_get("/api/tags").get("models", [])
                with ThreadPoolExecutor(8) as pool:
                    found = list(pool.map(caps, [m["name"] for m in tags]))
                models = [{"name": m["name"], "gb": round(m["size"] / 1e9, 1), "vision": "vision" in c,
                           "tools": "tools" in c, "text": "completion" in c or not c}  # unknown: assume it writes text
                          for m, c in zip(tags, found)]
                body = {"models": models, "default": bench.DEFAULT_CHAIN.split(","), "ram_gb": RAM_GB,
                        "modes": MODES}
            except OSError as e:
                body = {"error": f"Can't reach Ollama at {bench.OLLAMA}. Is the app running? ({e})"}
            self.reply(200, "application/json", json.dumps(body).encode())
        elif url.path == "/runs":
            self.reply(200, "application/json", json.dumps(bench.list_runs()).encode())
        elif url.path.startswith("/runs/") and url.path.endswith((".md", ".html")):
            rid, ext = url.path[len("/runs/"):].rsplit(".", 1)
            r = bench.load_run(rid)
            if not r:
                self.reply(404, "text/plain", b"no such run")
            elif ext == "md":  # a download
                self.reply(200, "text/markdown; charset=utf-8", report.to_markdown(r).encode(),
                           {"Content-Disposition": f'attachment; filename="{rid}.md"'})
            else:  # opens as a page; Print > Save as PDF works from there
                self.reply(200, "text/html; charset=utf-8", report.to_html(r).encode())
        elif url.path.startswith("/runs/"):
            r = bench.load_run(url.path[len("/runs/"):])
            if r:
                self.reply(200, "application/json", json.dumps(r, ensure_ascii=False).encode())
            else:
                self.reply(404, "text/plain", b"no such run")
        elif url.path.startswith("/images/"):
            name = url.path[len("/images/"):]
            path = bench.RUNS / "images" / name
            if re.fullmatch(r"[0-9a-f]{16}\.jpg", name) and path.is_file():
                self.reply(200, "image/jpeg", path.read_bytes())
            else:
                self.reply(404, "text/plain", b"no such image")
        elif url.path == "/run" and q.get("task"):
            self.run_mode(q)
        else:
            self.reply(404, "text/plain", b"not found")

    def do_POST(self):
        """Same as GET /run, but the settings come as JSON, so an image fits. Also /save."""
        size = int(self.headers.get("Content-Length") or 0)
        if urlparse(self.path).path == "/save" and 0 < size < 4096:
            return self.save_run(json.loads(self.rfile.read(size) or b"{}"))
        if urlparse(self.path).path != "/run" or not 0 < size <= MAX_BODY:
            return self.reply(413 if size > MAX_BODY else 404, "text/plain", b"bad request")
        try:
            q = json.loads(self.rfile.read(size))
        except ValueError:
            return self.reply(400, "text/plain", b"body isn't JSON")
        self.run_mode(q)

    def run_mode(self, q):
        """Stream run events to the page as server-sent events, one per line pair."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        split = lambda v: v if isinstance(v, list) else [x for x in str(v or "").split(",") if x]
        mode = q.get("mode", "blind")
        temp = lambda v: None if v in (None, "", "default") else float(v)  # "default": the model's own
        try:
            if mode not in MODES:
                raise ValueError(f"unknown mode {mode!r}")
            if mode == "vision" and q.get("format", "text") not in bench.FORMATS:
                raise ValueError(f"unknown format {q.get('format')!r}")
            r = bench.run(q.get("task", ""), split(q.get("chain")) or bench.DEFAULT_CHAIN.split(","), mode,
                          q.get("rule", ""), send,
                          images=q.get("images") or [q.get("image")], fmt=q.get("format", "text"),
                          combiner=q.get("combiner") or "granite4.2:8b", strategy=q.get("strategy", "batch"),
                          fields=q.get("fields") or [], rounds=int(q.get("rounds", 2) or 0),
                          ending=q.get("ending", "vote"), moderator=q.get("moderator") or "granite4.2:8b",
                          allow_self=bool(q.get("allow_self")),
                          temperature=temp(q.get("temperature", "default")),
                          temperatures={m: temp(v) for m, v in (q.get("temps") or {}).items()})
            send({"type": "done", **{k: v for k, v in r.items() if k != "trace"}})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed mid-run
        except Exception as e:
            send({"type": "error", "message": str(e)})

    def save_run(self, q):
        """Ask where to save a run with the Mac's own Save dialog, then write its report there.
        For browsers that can't show a save dialog themselves (Safari)."""
        rid = str(q.get("id", ""))
        r = bench.load_run(rid)
        if not r:
            return self.reply(404, "application/json", b'{"error": "no such run"}')
        script = ['-e', 'activate',  # brings the dialog to the front
                  '-e', f'POSIX path of (choose file name with prompt "Save this run (end the name with .html '
                        f'for a web page, otherwise it saves as Markdown)" default name "{rid}.md" '
                        f'default location (path to desktop folder))']
        try:
            done = subprocess.run(["osascript", *script], capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            return self.reply(500, "application/json", json.dumps({"error": str(e)}).encode())
        if done.returncode != 0:  # -128 is the Cancel button
            cancelled = "-128" in done.stderr
            return self.reply(200, "application/json",
                              json.dumps({"cancelled": True} if cancelled else {"error": done.stderr.strip()}).encode())
        path = Path(done.stdout.strip())
        if path.suffix.lower() not in (".md", ".html", ".htm"):
            path = path.with_name(path.name + ".md")
        text = report.to_html(r) if path.suffix.lower() in (".html", ".htm") else report.to_markdown(r)
        path.write_text(text)
        self.reply(200, "application/json", json.dumps({"path": str(path)}).encode())

    def reply(self, code, ctype, body, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Local LLM Bench on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
