"""
download_sources.py — fetch every enabled source from the registry.

What this does
--------------
Reads sources_registry.SOURCES, walks each enabled entry, and downloads its
files into data/raw/<source_key>/. The downloader is deliberately dumb: it
just gets the bytes onto disk. Parsing happens in a separate step (parsers/)
so a download failure on one source doesn't block ingest of the others, and
so re-parsing during prompt iteration doesn't re-hit the network.

Why HTTPS over `requests` rather than git for everything
--------------------------------------------------------
Most of our sources are individual JSON or HTML files. Cloning a whole repo
to get two files wastes bandwidth and brittle-ifies the script. For sources
that *are* whole repos (rare in our registry), prefix the URL with `git+`.

Idempotency
-----------
If a file is already present and not corrupt, we skip it. Pass --force to
re-download. This makes it safe to run repeatedly while debugging parsers.

Politeness
----------
We send a real User-Agent and rate-limit to one request per second per host.
Internet Archive and similar mirrors are gracious to projects that play nice;
they can also throttle aggressively when they aren't.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

import config
from sources_registry import SOURCES, Source


RAW_DIR = config.DATA_DIR / "raw"
USER_AGENT = (
    "GitaAdvisor/0.2 (Advaita-Vedanta research project; "
    "contact: <add your email here>)"
)

# Per-host minimum interval in seconds
MIN_INTERVAL = 1.0


def _filename_for_url(url: str) -> str:
    """Derive a sensible local filename from a URL."""
    parsed = urlparse(url)
    name = Path(parsed.path).name or "index.html"
    # archive.org sometimes serves djvu.txt with no extension on the URL;
    # keep what's there.
    return name


def _is_git_url(url: str) -> bool:
    return url.startswith("git+")


_last_request_time: dict = defaultdict(float)


def _polite_get(url: str) -> requests.Response:
    """GET with rate limiting per host."""
    host = urlparse(url).netloc
    elapsed = time.time() - _last_request_time[host]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_time[host] = time.time()
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True)


def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a single URL to dest. Returns True if a download happened
    (vs being skipped because already present)."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    with _polite_get(url) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with tmp.open("wb") as out, tqdm(
            total=total, unit="B", unit_scale=True, leave=False, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                out.write(chunk)
                bar.update(len(chunk))

    tmp.replace(dest)
    return True


def _clone_git(url: str, dest_dir: Path, force: bool = False) -> bool:
    """Clone a git repo (URL prefixed with 'git+') into dest_dir. Returns
    True if a clone happened."""
    real_url = url[len("git+"):]
    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        return False
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", real_url, str(dest_dir)],
        check=True,
    )
    return True


def download_source(src: Source, force: bool = False) -> dict:
    """Download all URLs for one source. Returns a small report dict."""
    target = RAW_DIR / src.key
    report = {"key": src.key, "ok": 0, "skipped": 0, "failed": []}

    if not src.urls:
        report["failed"].append("no URLs in registry entry")
        return report

    for url in src.urls:
        if not url:
            continue
        try:
            if _is_git_url(url):
                changed = _clone_git(url, target, force=force)
            else:
                fname = _filename_for_url(url)
                changed = _download_file(url, target / fname, force=force)
            if changed:
                report["ok"] += 1
            else:
                report["skipped"] += 1
        except Exception as e:
            report["failed"].append(f"{url}: {e}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Download all enabled sources from the registry.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files exist.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only download these source keys.")
    args = ap.parse_args()

    enabled = [s for s in SOURCES if s.enabled]
    if args.only:
        enabled = [s for s in enabled if s.key in set(args.only)]
    if not enabled:
        print("No enabled sources match. Edit sources_registry.py to enable some.")
        sys.exit(1)

    print(f"Downloading {len(enabled)} sources to {RAW_DIR}")
    print(f"User-Agent: {USER_AGENT}")
    print()

    any_failed = False
    for src in enabled:
        print(f"━━━ {src.key} — {src.name}")
        print(f"    license={src.license}  tier={src.tier}  parser={src.parser}")
        if src.translator:
            year = f", {src.year}" if src.year else ""
            print(f"    translator: {src.translator}{year}")

        report = download_source(src, force=args.force)
        if report["failed"]:
            any_failed = True
            for f in report["failed"]:
                print(f"    [FAIL] {f}")
        print(f"    downloaded={report['ok']}  cached={report['skipped']}")
        print()

    if any_failed:
        print("Some sources failed. Re-run with the network available, or "
              "edit the URL in sources_registry.py if a mirror has moved.")
        sys.exit(2)
    print("All enabled sources are now on disk under data/raw/.")
    print("Next: python ingest_corpus.py")


if __name__ == "__main__":
    main()
