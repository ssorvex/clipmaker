"""A small web front end for clipmaker, so a link can be pasted from anywhere.

The interface cannot do the work: downloading, transcribing and rendering need
this machine's graphics card, which is why there is a server here at all
rather than a copy of the tool on the laptop. This serves a page, runs
clipmaker as a subprocess, streams its output back, and hands the finished
clips over for preview and posting.

Standard library only. One person using one machine does not need a web
framework, and a tool that cuts video should not grow a dependency tree to
show a progress bar.

    python server.py                  # http://127.0.0.1:8777, this PC only
    python server.py --host 0.0.0.0   # reachable from the home network
    python server.py --port 9000

Set CLIPMAKER_TOKEN to require ?t=<token> on every request. Worth doing the
moment the server listens on anything but 127.0.0.1, and not a substitute for
Cloudflare Access once it is on the internet.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CLIPS = ROOT / "clips"

# How many log lines to keep per job. Enough to see what happened, bounded so
# a long run cannot grow without limit.
LOG_LINES = 400

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

TOKEN = (os.environ.get("CLIPMAKER_TOKEN") or "").strip()


def has_llm_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# --------------------------------------------------------------------------- #
# running a job
# --------------------------------------------------------------------------- #

def _set(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)


def _log(job_id: str, line: str) -> None:
    with _lock:
        lines = _jobs[job_id]["lines"]
        lines.append(line)
        del lines[:-LOG_LINES]


def _phone_copy(out_dir: Path, job_id: str) -> None:
    """Best effort: get the clips to the phone folder as the CLI does.

    A failure here is not worth failing a finished render over, so it only
    shows up in the log.
    """
    ps = shutil.which("powershell") or shutil.which("pwsh")
    script = ROOT / "publish.ps1"
    if not ps or not script.is_file():
        return
    try:
        done = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-From", str(out_dir)],
            capture_output=True, text=True, timeout=300,
        )
        for line in (done.stdout or "").splitlines():
            if line.strip():
                _log(job_id, line.rstrip())
    except Exception as exc:               # noqa: BLE001 - reported, never raised
        _log(job_id, f"  (phone folder copy failed: {exc})")


def _run_job(job_id: str, opts: dict) -> None:
    out_dir = CLIPS / time.strftime("%Y-%m-%d_%H%M%S")
    _set(job_id, out_dir=str(out_dir), state="running")

    cmd = [
        sys.executable, "-u", str(ROOT / "clipmaker.py"), "make", opts["url"],
        "--out", str(out_dir),
        "--clips", str(opts["clips"]),
        "--min-len", str(opts["min_len"]),
        "--max-len", str(opts["max_len"]),
        "--crop", opts["crop"],
    ]
    if opts.get("direction"):
        cmd += ["--direction", opts["direction"]]
    if opts.get("cookies"):
        cmd += ["--cookies-from-browser", opts["cookies"]]
    if not has_llm_key():
        # Without a key the local planner picks the moments. It scores loudness
        # and pace, which is not the same thing as interesting - the page says
        # so, rather than quietly shipping weaker clips.
        cmd.append("--no-llm")

    _log(job_id, "> " + " ".join(cmd[2:]))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
    except Exception as exc:               # noqa: BLE001
        _set(job_id, state="failed", error=str(exc))
        return

    with _lock:
        _jobs[job_id]["pid"] = proc.pid

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _log(job_id, line)
    code = proc.wait()

    if code == 0:
        _phone_copy(out_dir, job_id)
        _set(job_id, state="done")
    else:
        _set(job_id, state="failed", error=f"clipmaker exited with code {code}")


def start_job(opts: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "state": "starting", "lines": [], "error": "",
            "out_dir": "", "url": opts["url"], "started": time.time(), "pid": 0,
        }
    threading.Thread(target=_run_job, args=(job_id, opts), daemon=True).start()
    return job_id


# --------------------------------------------------------------------------- #
# reading what a job produced
# --------------------------------------------------------------------------- #

def job_clips(job_id: str) -> list[dict]:
    with _lock:
        job = _jobs.get(job_id)
        out = job.get("out_dir") if job else ""
    if not out:
        return []
    out_dir = Path(out)
    if not out_dir.is_dir():
        return []

    found = []
    for n, folder in enumerate(sorted(p for p in out_dir.iterdir() if p.is_dir()), start=1):
        clip = folder / "clip.mp4"
        meta = folder / "meta.json"
        if not clip.is_file():
            continue
        plan = {}
        if meta.is_file():
            try:
                plan = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("plan") or {}
            except (OSError, json.JSONDecodeError):
                plan = {}
        found.append({
            "n": n,
            "title": plan.get("title") or folder.name,
            "description": plan.get("description") or "",
            "hashtags": plan.get("hashtags") or [],
            "start": plan.get("start") or 0,
            "duration": round(float(plan.get("duration") or 0), 1),
            "size_mb": round(clip.stat().st_size / (1024 * 1024), 1),
            "why": plan.get("why") or "",
        })
    return found


def _clip_path(job_id: str, n: int) -> Path | None:
    with _lock:
        job = _jobs.get(job_id)
        out = job.get("out_dir") if job else ""
    if not out:
        return None
    folders = sorted(p for p in Path(out).iterdir() if p.is_dir()) if Path(out).is_dir() else []
    if 1 <= n <= len(folders):
        clip = folders[n - 1] / "clip.mp4"
        return clip if clip.is_file() else None
    return None


# --------------------------------------------------------------------------- #
# posting
# --------------------------------------------------------------------------- #

def do_post(job_id: str, n: int, platforms: list[str], title: str, text: str) -> dict:
    from cm import woop

    clip = _clip_path(job_id, n)
    if clip is None:
        raise woop.WoopError("that clip is not on disk any more")

    client = woop.Woop()
    projects = client.projects()
    if not projects:
        raise woop.WoopError("no project at WoopSocial yet - make one in its dashboard")
    project_id = str(projects[0].get("id") or "")

    wanted = {p.strip().lower() for p in platforms if p.strip()}
    accounts = [a for a in client.accounts(project_id) if a["name"] in wanted]
    if not accounts:
        raise woop.WoopError(
            "none of " + ", ".join(sorted(wanted)) + " is connected at WoopSocial"
        )

    media_id = client.upload(clip, project_id)
    body = woop.build_body(media_id=media_id, text=text, title=title, accounts=accounts)
    answer = client.publish(body)
    return {
        "post_id": str(answer.get("id") or answer.get("postId") or ""),
        "results": woop.read_results(answer),
    }


def account_status() -> dict:
    """What the Post button can actually reach, said plainly."""
    try:
        from cm import woop
        client = woop.Woop()
        projects = client.projects()
        project_id = str(projects[0].get("id") or "") if projects else ""
        accounts = client.accounts(project_id)
        return {
            "ok": True,
            "connected": [a["name"] for a in accounts],
            "labels": {a["name"]: a["label"] for a in accounts},
        }
    except Exception as exc:               # noqa: BLE001 - the message is the answer
        return {"ok": False, "connected": [], "reason": str(exc)}


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "clipmaker"
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------- #

    def _authed(self, query: dict) -> bool:
        if not TOKEN:
            return True
        given = (query.get("t") or [""])[0] or self.headers.get("X-Clipmaker-Token", "")
        return given == TOKEN

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):       # quieter: one line, no client noise
        sys.stderr.write("  %s\n" % (fmt % args))

    # -- video, with range support so previews can seek --------------------- #

    def _serve_video(self, path: Path) -> None:
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)", rng) if rng else None

        if not match:
            with path.open("rb") as fh:
                body = fh.read()
            self._send(200, body, "video/mp4", {"Accept-Ranges": "bytes"})
            return

        start = int(match.group(1)) if match.group(1) else 0
        end = int(match.group(2)) if match.group(2) else size - 1
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        with path.open("rb") as fh:
            fh.seek(start)
            body = fh.read(end - start + 1)
        self._send(206, body, "video/mp4", {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
        })

    # -- routes ------------------------------------------------------------- #

    def do_GET(self):                        # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path

        if not self._authed(query):
            self._json({"error": "bad or missing token"}, 403)
            return

        if route in ("/", "/index.html"):
            page = WEB / "index.html"
            if not page.is_file():
                self._send(500, b"web/index.html is missing", "text/plain; charset=utf-8")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if route == "/api/env":
            return self._json({
                "llm": has_llm_key(),
                "token": bool(TOKEN),
                "accounts": account_status(),
            })

        if route == "/api/job":
            job_id = (query.get("id") or [""])[0]
            with _lock:
                job = dict(_jobs.get(job_id) or {})
            if not job:
                return self._json({"error": "no such job"}, 404)
            job["lines"] = list(job.get("lines") or [])
            job["clips"] = job_clips(job_id) if job.get("state") == "done" else []
            return self._json(job)

        if route == "/api/jobs":
            with _lock:
                rows = [
                    {"id": j["id"], "url": j["url"], "state": j["state"],
                     "started": j["started"]}
                    for j in _jobs.values()
                ]
            rows.sort(key=lambda r: r["started"], reverse=True)
            return self._json({"jobs": rows[:20]})

        if route == "/api/video":
            job_id = (query.get("id") or [""])[0]
            try:
                n = int((query.get("n") or ["0"])[0])
            except ValueError:
                n = 0
            clip = _clip_path(job_id, n)
            if clip is None:
                return self._json({"error": "no such clip"}, 404)
            return self._serve_video(clip)

        self._json({"error": "no such route"}, 404)

    def do_HEAD(self):                       # noqa: N802
        self.do_GET()

    def do_POST(self):                       # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authed(query):
            self._json({"error": "bad or missing token"}, 403)
            return

        payload = self._read_json()

        if parsed.path == "/api/start":
            url = str(payload.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                return self._json({"error": "that does not look like a link"}, 400)
            opts = {
                "url": url,
                "clips": max(1, min(int(payload.get("clips") or 3), 10)),
                "min_len": max(5, min(float(payload.get("min_len") or 30), 120)),
                "max_len": max(10, min(float(payload.get("max_len") or 60), 180)),
                "crop": payload.get("crop") if payload.get("crop") in ("fill", "blur", "pad") else "fill",
                "direction": str(payload.get("direction") or "").strip()[:500],
                "cookies": payload.get("cookies") if payload.get("cookies") in
                           ("chrome", "edge", "firefox", "brave") else "",
            }
            return self._json({"id": start_job(opts)})

        if parsed.path == "/api/post":
            try:
                out = do_post(
                    str(payload.get("id") or ""),
                    int(payload.get("n") or 0),
                    list(payload.get("platforms") or []),
                    str(payload.get("title") or "")[:100],
                    str(payload.get("text") or "")[:2200],
                )
            except Exception as exc:         # noqa: BLE001 - the message is the answer
                return self._json({"error": str(exc)}, 400)
            return self._json(out)

        self._json({"error": "no such route"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser(description="clipmaker's web front end")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default, this PC only) or 0.0.0.0 (home network)")
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()

    mimetypes.add_type("video/mp4", ".mp4")
    CLIPS.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print()
    print(f"  clipmaker web   http://{shown}:{args.port}")
    if args.host == "0.0.0.0":
        print("  reachable from the home network on this PC's IP address")
        if not TOKEN:
            print("  ! no CLIPMAKER_TOKEN set, so anyone on the network can use it")
    print(f"  clip picking    {'Claude (API key found)' if has_llm_key() else 'local planner (no API key)'}")
    print("  stop with Ctrl+C")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
