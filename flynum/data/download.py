"""Download MaleCNS v1.0 flat-connectome files.

Hard rules implemented here
---------------------------
* Every request goes through the local proxy (default ``http://127.0.0.1:7897``).
* **A single file larger than 5 GiB is never downloaded silently.**  The
  downloader raises :class:`ApprovalRequired` and prints an explicit approval
  request instead.  Pass ``allow_over_limit=True`` (or ``--approve`` on the CLI)
  only after the user has approved.
* Downloads resume from a ``.part`` file and are verified by byte size.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .. import paths

CHUNK = 4 * 1024 * 1024  # 4 MiB
ProgressFn = Callable[[int, int], None]


class ApprovalRequired(RuntimeError):
    """Raised when a download exceeds the size the user pre-approved."""


@dataclass
class FileInfo:
    key: str
    name: str
    url: str
    size_bytes: int
    size_mb: float
    over_limit: bool

    def line(self) -> str:
        mark = "  <-- REQUIRES APPROVAL (>5 GiB)" if self.over_limit else ""
        return f"{self.name:<58} {self.size_mb:9.2f} MB{mark}"


# --------------------------------------------------------------------------- #
def _opener(proxy: str | None = None) -> urllib.request.OpenerDirector:
    proxy = proxy or paths.PROXY
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    return urllib.request.build_opener(handler)


def head_size(url: str, proxy: str | None = None, timeout: int = 60) -> tuple[int, bool]:
    """Return ``(content_length, accept_ranges)`` for a URL."""
    req = urllib.request.Request(url, method="HEAD")
    with _opener(proxy).open(req, timeout=timeout) as resp:
        size = int(resp.headers["Content-Length"])
        ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    return size, ranges


def probe(key: str, proxy: str | None = None) -> FileInfo:
    """HEAD a MaleCNS file and classify it against the approval limit."""
    try:
        name = paths.REQUIRED_FILES[key]
    except KeyError:
        name = paths.OPTIONAL_FILES[key]
    url = paths.MALECNS_BASE + name
    size, _ = head_size(url, proxy)
    return FileInfo(
        key=key,
        name=name,
        url=url,
        size_bytes=size,
        size_mb=size / 1024**2,
        over_limit=size > paths.SIZE_APPROVAL_LIMIT,
    )


def probe_all(proxy: str | None = None) -> list[FileInfo]:
    """HEAD every known file (required first, then optional)."""
    out = []
    for key in list(paths.REQUIRED_FILES) + list(paths.OPTIONAL_FILES):
        try:
            out.append(probe(key, proxy))
        except Exception as exc:  # network hiccup should not kill the listing
            out.append(
                FileInfo(
                    key=key,
                    name=f"{key} (unreachable: {exc})",
                    url="",
                    size_bytes=-1,
                    size_mb=float("nan"),
                    over_limit=False,
                )
            )
    return out


# --------------------------------------------------------------------------- #
def verify_file(path: Path, expected_size: int | None = None) -> bool:
    """Cheap integrity check: exists and has the expected byte count."""
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return path.stat().st_size > 0


def sha256_file(path: Path, progress: ProgressFn | None = None) -> str:
    h = hashlib.sha256()
    done = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, path.stat().st_size)
    return h.hexdigest()


def download_file(
    key: str,
    *,
    proxy: str | None = None,
    force: bool = False,
    allow_over_limit: bool = False,
    progress: ProgressFn | None = None,
    logger=None,
) -> Path:
    """Download one MaleCNS file into ``data/raw`` (resumable, verified).

    Raises
    ------
    ApprovalRequired
        If the file exceeds the 5 GiB limit and ``allow_over_limit`` is False.
    """
    info = probe(key, proxy)
    if info.over_limit and not allow_over_limit:
        raise ApprovalRequired(
            f"\n{'=' * 78}\n"
            f"APPROVAL NEEDED: {info.name}\n"
            f"  size: {info.size_mb:,.1f} MB ({info.size_bytes / 1024**3:.2f} GiB) "
            f"> 5 GiB limit\n"
            f"  url : {info.url}\n"
            f"  Re-run with allow_over_limit=True (or --approve) after the user "
            f"approves this download.\n{'=' * 78}\n"
        )

    dest = paths.raw_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force and verify_file(dest, info.size_bytes):
        if logger:
            logger.info("[skip] %s already present (%.1f MB)", info.name, info.size_mb)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() and not force else 0
    if have >= info.size_bytes:
        have = 0
        part.unlink(missing_ok=True)

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(info.url, headers=headers)
    if logger:
        logger.info(
            "[get ] %s (%.1f MB)%s",
            info.name,
            info.size_mb,
            f" resume at {have / 1024**2:.1f} MB" if have else "",
        )

    t0 = time.time()
    mode = "ab" if have else "wb"
    with _opener(proxy).open(req, timeout=120) as resp, part.open(mode) as fh:
        done = have
        last = t0
        while chunk := resp.read(CHUNK):
            fh.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, info.size_bytes)
            now = time.time()
            if logger and now - last > 10:
                rate = (done - have) / max(now - t0, 1e-9) / 1024**2
                logger.info(
                    "       %5.1f%%  %7.1f/%7.1f MB  %5.2f MB/s",
                    100 * done / info.size_bytes,
                    done / 1024**2,
                    info.size_mb,
                    rate,
                )
                last = now

    if not verify_file(part, info.size_bytes):
        got = part.stat().st_size if part.exists() else 0
        raise IOError(
            f"size mismatch for {info.name}: got {got} bytes, expected {info.size_bytes}"
        )

    shutil.move(str(part), str(dest))
    if logger:
        logger.info(
            "[ok  ] %s (%.1f MB in %.1fs)",
            info.name,
            info.size_mb,
            time.time() - t0,
        )
    return dest


def download_required(
    *, proxy: str | None = None, force: bool = False, logger=None
) -> dict[str, Path]:
    """Download the three files needed for the core experiment (~1.06 GB)."""
    out: dict[str, Path] = {}
    for key in paths.REQUIRED_FILES:
        out[key] = download_file(key, proxy=proxy, force=force, logger=logger)
    return out


# --------------------------------------------------------------------------- #
def write_download_manifest(logger=None) -> Path:
    """Record sizes/hashes of every present raw file (audit trail)."""
    entries = []
    for key in list(paths.REQUIRED_FILES) + list(paths.OPTIONAL_FILES):
        p = paths.raw_path(key)
        if p.exists():
            entries.append(
                {
                    "key": key,
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    out = paths.DATA_PROCESSED / "raw_manifest.json"
    out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    if logger:
        logger.info("wrote %s (%d files)", out, len(entries))
    return out
