"""Post a finished clip to TikTok, Instagram and YouTube through WoopSocial.

Why a publishing service instead of the platforms directly: TikTok and YouTube
both lock an unaudited app's uploads to private, and YouTube's lock is
permanent per video -- it cannot be lifted in Studio afterwards and there is
no appeal. WoopSocial already holds those audits and the OAuth connections, so
a clip can go out publicly on the day it was cut. Their free plan covers two
accounts, which is why TikTok and Instagram are the default pair: YouTube is
the one platform that is quick to post by hand.

No credential is stored here. The API key is read from the environment and
sent in one Authorization header; the social accounts live at WoopSocial's
end, so this never sees a social password.

Standard library only, deliberately: posting should not add a dependency to a
tool whose actual job is cutting video.
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE_URL = "https://api.woopsocial.com/v1"
TIMEOUT = 30
UPLOAD_TIMEOUT = 900

# Their single-request upload is good to 100 MB; past that they want a chunked
# session. A 60-second 1080x1920 clip is well under, so the limit is reported
# rather than worked around.
MAX_UPLOAD = 100 * 1024 * 1024

# Their identifiers are upper case; everything in clipmaker is lower case.
PLATFORMS = {
    "youtube": "YOUTUBE",
    "tiktok": "TIKTOK",
    "instagram": "INSTAGRAM",
    "facebook": "FACEBOOK",
    "x": "X",
    "threads": "THREADS",
    "linkedin": "LINKEDIN",
}
BY_ID = {v: k for k, v in PLATFORMS.items()}

DEFAULT_PLATFORMS = ("tiktok", "instagram")


class WoopError(Exception):
    pass


def api_key() -> str:
    key = (os.environ.get("WOOPSOCIAL_API_KEY") or "").strip()
    if not key:
        raise WoopError(
            "WOOPSOCIAL_API_KEY is not set.\n"
            "  1. Make a free account at https://woopsocial.com\n"
            "  2. Connect TikTok and Instagram on its accounts page\n"
            "  3. Copy the API key from the dashboard, then in PowerShell:\n"
            '       setx WOOPSOCIAL_API_KEY "paste-it-here"\n'
            "     and open a new terminal."
        )
    return key


def _multipart(path: Path) -> tuple[str, bytes]:
    """Encode one file as multipart/form-data under the field name 'file'."""
    boundary = "----clipmaker" + uuid.uuid4().hex
    guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + path.name + '"\r\n'
        "Content-Type: " + guessed + "\r\n\r\n"
    ).encode("utf-8")
    tail = ("\r\n--" + boundary + "--\r\n").encode("utf-8")
    return "multipart/form-data; boundary=" + boundary, head + path.read_bytes() + tail


class Woop:
    def __init__(self, key: str = "", *, base_url: str = BASE_URL):
        self._key = key or api_key()
        self.base = base_url.rstrip("/")

    def _call(self, method: str, path: str, *, params: dict | None = None,
              body: dict | None = None, upload: Path | None = None,
              timeout: int = TIMEOUT):
        url = self.base + path
        if params:
            kept = {k: v for k, v in params.items() if v}
            if kept:
                url += "?" + urllib.parse.urlencode(kept)

        headers = {"Authorization": "Bearer " + self._key, "Accept": "application/json"}
        data = None
        if upload is not None:
            headers["Content-Type"], data = _multipart(upload)
        elif body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise WoopError(self._explain(exc.code, raw)) from None
        except urllib.error.URLError as exc:
            raise WoopError("could not reach WoopSocial: " + str(exc.reason)) from None

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise WoopError("WoopSocial returned something that is not JSON: "
                            + raw[:300]) from None

    @staticmethod
    def _explain(status: int, raw: str) -> str:
        detail = raw[:400]
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("error")
                             or payload.get("detail") or detail)
        except json.JSONDecodeError:
            pass
        if status in (401, 403):
            return "WoopSocial rejected the API key (" + str(status) + "): " + detail
        if status == 402:
            return "WoopSocial says the plan does not cover this: " + detail
        if status == 429:
            return "WoopSocial rate limit or daily allowance reached: " + detail
        return "WoopSocial refused the request (" + str(status) + "): " + detail

    # ---- reading -----------------------------------------------------------

    @staticmethod
    def _rows(got) -> list[dict]:
        if isinstance(got, list):
            return got
        if isinstance(got, dict):
            for field in ("data", "projects", "socialAccounts"):
                if isinstance(got.get(field), list):
                    return got[field]
        return []

    def projects(self) -> list[dict]:
        return self._rows(self._call("GET", "/projects"))

    def accounts(self, project_id: str = "") -> list[dict]:
        rows = self._rows(self._call("GET", "/social-accounts",
                                     params={"projectId": project_id}))
        out = []
        for row in rows:
            name = BY_ID.get(str(row.get("platform") or "").upper())
            if name and row.get("id"):
                out.append({
                    "name": name,
                    "id": row["id"],
                    "platform": row["platform"],
                    "label": row.get("username") or row.get("name") or "",
                })
        return out

    # ---- posting -----------------------------------------------------------

    def upload(self, video: Path, project_id: str) -> str:
        size = video.stat().st_size
        if size > MAX_UPLOAD:
            raise WoopError(
                video.name + " is " + str(size // (1024 * 1024)) + " MB and WoopSocial "
                "takes 100 MB in one piece. Re-render it shorter or smaller."
            )
        got = self._call("POST", "/media", params={"projectId": project_id},
                         upload=video, timeout=UPLOAD_TIMEOUT)
        media_id = str(got.get("id") or got.get("mediaId") or "")
        if not media_id:
            raise WoopError("WoopSocial took the clip but returned no media id.")
        return media_id

    def publish(self, body: dict) -> dict:
        return self._call("POST", "/posts", body=body, timeout=UPLOAD_TIMEOUT)

    def post_status(self, post_id: str) -> dict:
        return self._call("GET", "/posts/" + urllib.parse.quote(str(post_id), safe=""))


def build_body(*, media_id: str, text: str, title: str, accounts: list[dict]) -> dict:
    """One parent post, one child per destination.

    Each platform rejects a child missing one of its own required fields, so
    every destination gets its full set rather than a shared default that
    happens to suit YouTube.
    """
    children: list[dict] = []
    for acct in accounts:
        child = {"platform": acct["platform"], "socialAccountId": acct["id"]}
        if acct["name"] == "youtube":
            # An untitled video is refused outright, not defaulted.
            child["title"] = title[:100]
            child["privacy"] = "public"
        elif acct["name"] == "tiktok":
            child["postType"] = "VIDEO"
            child["privacyLevel"] = "PUBLIC_TO_EVERYONE"
            child["allowComment"] = True
            child["allowDuet"] = True
            child["allowStitch"] = True
            # Declarations about paid partnerships, not preferences. False is
            # the true answer for an ordinary clip, so it is stated here
            # rather than left for the service to assume either way.
            child["isYourBrand"] = False
            child["isBrandedContent"] = False
            child["autoAddMusic"] = False
        elif acct["name"] in ("instagram", "facebook"):
            child["postType"] = "REEL"      # vertical, so a Reel not a feed post
        children.append(child)

    if not children:
        raise WoopError("none of the chosen platforms is connected at WoopSocial")

    return {
        "content": [{"text": text,
                     "media": [{"type": "MEDIA_LIBRARY", "mediaId": media_id}]}],
        "schedule": {"type": "PUBLISH_NOW"},
        "socialAccounts": children,
    }


def caption(plan: dict, *, limit: int = 2200) -> str:
    """Description plus hashtags, which is what every vertical platform wants."""
    text = (plan.get("description") or plan.get("title") or "").strip()
    tags = " ".join("#" + str(t).lstrip("#") for t in (plan.get("hashtags") or []))
    joined = (text + "\n\n" + tags).strip() if tags else text
    return joined[:limit]


def read_results(payload: dict) -> list[dict]:
    """Flatten the per-destination outcome of one post into plain rows."""
    children = (
        payload.get("socialAccountPosts")
        or payload.get("socialAccounts")
        or payload.get("children")
        or []
    )
    rows = []
    for child in children:
        if not isinstance(child, dict):
            continue
        rows.append({
            "platform": BY_ID.get(str(child.get("platform") or "").upper(),
                                  str(child.get("platform") or "?")),
            "status": str(child.get("deliveryStatus") or child.get("status") or "?").upper(),
            "url": str(child.get("externalPostUrl") or child.get("permalink")
                       or child.get("url") or ""),
            "error": str(child.get("errorMessage") or child.get("error")
                         or child.get("failureReason") or ""),
        })
    return rows
