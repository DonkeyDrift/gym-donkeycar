"""Managed installation and updates for the Donkey Car simulator.

The Donkey Unity simulator ships as a GitHub release asset: ``DonkeySimMac.zip`` on macOS,
``DonkeySimWin.zip`` on Windows and ``DonkeySimLinux.zip`` on Linux.  This module downloads,
verifies and installs those assets into a managed directory and keeps the most recent
versions available for quick switching.

The managed directory (``~/.donkeycar/simulators`` by default) contains::

    versions/       installed simulator versions, one directory per tag
    downloads/      downloaded archives and their ``*.part`` partial files
    current.json    record of the version currently selected for use
    staged.json     a staged update waiting to be applied
    last_check.json cache of the most recent GitHub release lookup (24 h TTL)
    updates.lock    cross-process lock guarding mutating operations

Only the standard library is used.  The GitHub API is queried with ``urllib`` exactly once
per attempt (no automatic retries); HTTP 403 / 429 / abuse-detection responses raise
``SimulatorRateLimitError``.  Archives must carry a ``sha256:<hex>`` digest on their release
asset, or the download is refused.  Downloads resume through HTTP ``Range`` requests, are
verified against the digest and the asset size, and are moved into place atomically.
Extraction rejects absolute paths, ``..`` components and symbolic links.  A pending update
is staged under ``.staging-<safe-tag>-<uuid>`` and becomes live through
``apply_staged_update()``, which renames the staging directory into ``versions/`` and
atomically rewrites ``current.json``.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import stat
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if os.name == "nt":
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

DEFAULT_SIMULATOR_UPDATE_REPO = "DonkeyDrift/gym-donkeycar"

_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
_CHUNK_SIZE = 64 * 1024
_API_LATEST_URL = "https://api.github.com/repos/{repo}/releases/latest"
_API_RELEASES_URL = "https://api.github.com/repos/{repo}/releases?per_page=30"
_USER_AGENT = "gym-donkeycar-simulator-updater/1.0 (+https://github.com/DonkeyDrift/gym-donkeycar)"
_VERSION_METADATA_NAME = "version.json"

_PLATFORM_ASSET_NAMES = {
    "darwin": "DonkeySimMac.zip",
    "win32": "DonkeySimWin.zip",
    "linux": "DonkeySimLinux.zip",
    "linux2": "DonkeySimLinux.zip",
}

_SHA256_DIGEST_RE = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_TAG_LENGTH = 120


class SimulatorUpdateError(Exception):
    """Base class for all simulator update failures."""


class SimulatorDownloadError(SimulatorUpdateError):
    """Downloading or verifying a simulator asset failed."""


class SimulatorRateLimitError(SimulatorDownloadError):
    """GitHub refused the request (HTTP 403, rate limit, or abuse detection)."""


@dataclass
class SimulatorRelease:
    """A published simulator release and the asset to download for this platform."""

    tag: str
    asset_name: str
    asset_url: str
    asset_size: int
    asset_sha256: str
    published_at: str


@dataclass
class InstalledSimulator:
    """A simulator version installed in the managed directory."""

    tag: str
    executable: str
    asset_name: str
    asset_sha256: str
    asset_size: int
    installed_at: str


@dataclass
class UpdateCheck:
    """Result of checking the installed version against the latest release."""

    installed: Optional[InstalledSimulator]
    latest: Optional[SimulatorRelease]
    update_available: bool
    checked_at: str


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _silent_unlink(path: Path) -> None:
    try:
        os.unlink(str(path))
    except OSError:
        pass


def sanitize_tag(tag: Any) -> str:
    """Return ``tag`` as a directory name that can never escape the managed root."""
    if not isinstance(tag, str) or not tag.strip():
        raise SimulatorUpdateError("invalid simulator version tag: %r" % (tag,))
    safe = _SAFE_TAG_RE.sub("_", tag).lstrip(".")
    safe = safe.rstrip(". ")[:_MAX_TAG_LENGTH]
    return safe or "unknown"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path`` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / ("%s.tmp-%s" % (path.name, uuid.uuid4().hex))
    try:
        with open(str(tmp_path), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_path), str(path))
    except OSError as exc:
        _silent_unlink(tmp_path)
        raise SimulatorUpdateError("cannot write %s: %s" % (path, exc))


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load a managed JSON file, raising a clear error instead of crashing on corruption."""
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SimulatorUpdateError("cannot read %s: %s" % (path, exc))
    if not isinstance(data, dict):
        raise SimulatorUpdateError("invalid format in %s: expected a JSON object" % path)
    return data


def _installed_to_dict(record: InstalledSimulator) -> Dict[str, Any]:
    return {
        "tag": record.tag,
        "executable": record.executable,
        "asset_name": record.asset_name,
        "asset_sha256": record.asset_sha256,
        "asset_size": record.asset_size,
        "installed_at": record.installed_at,
    }


def _release_to_dict(release: SimulatorRelease) -> Dict[str, Any]:
    return {
        "tag": release.tag,
        "asset_name": release.asset_name,
        "asset_url": release.asset_url,
        "asset_size": release.asset_size,
        "asset_sha256": release.asset_sha256,
        "published_at": release.published_at,
    }


def _release_from_dict(data: Dict[str, Any]) -> SimulatorRelease:
    return SimulatorRelease(
        tag=data["tag"],
        asset_name=data["asset_name"],
        asset_url=data["asset_url"],
        asset_size=int(data["asset_size"]),
        asset_sha256=data["asset_sha256"],
        published_at=data["published_at"],
    )


def _parse_sha256_digest(value: Any, asset_name: str) -> str:
    """Return the hex digest from a ``sha256:<hex>`` value, refusing anything else."""
    match = _SHA256_DIGEST_RE.match(value or "")
    if match:
        return match.group(1)
    raise SimulatorDownloadError("asset %r has no valid 'sha256:<hex>' digest; refusing to download" % asset_name)


def _pid_exists_windows(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    error_invalid_parameter = 87
    win_dll = getattr(ctypes, "WinDLL")
    get_last_error = getattr(ctypes, "get_last_error")
    kernel32 = win_dll("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if handle:
        kernel32.CloseHandle(handle)
        return True
    if get_last_error() == error_invalid_parameter:
        return False
    return True


def _pid_exists(pid: Optional[int]) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    if os.name == "nt":
        return _pid_exists_windows(int(pid))
    try:
        waited = os.waitpid(int(pid), os.WNOHANG)
    except ChildProcessError:
        waited = (0, 0)
    if waited != (0, 0):
        # A child of this process that already exited: reap it and report it gone.
        # kill(pid, 0) would keep succeeding for an unreaped zombie forever.
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def wait_for_pid(pid: int, timeout: Optional[float] = None) -> bool:
    """Block until process ``pid`` exits.

    Return ``True`` once the process is gone and ``False`` if it is still running when the
    optional ``timeout`` (seconds) elapses.  With ``timeout=None`` this waits indefinitely.
    If ``pid`` is a child of this process it is reaped once it exits, so the wait also
    works for child processes that nobody else waits on.
    """
    deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
    while True:
        if not _pid_exists(pid):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def resolve_managed_executable(root: Optional[str] = None) -> Optional[str]:
    """Return the executable path recorded in the managed ``current.json``, or ``None``.

    ``None`` is returned when the managed root does not exist yet, when no version is
    selected, or when the recorded executable has disappeared from disk.  A corrupted or
    malformed ``current.json`` raises :class:`SimulatorUpdateError`.
    """
    root_path = _managed_root(root)
    current_path = root_path / "current.json"
    if not current_path.exists():
        return None
    data = _load_json_file(current_path)
    executable = data.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        raise SimulatorUpdateError("current.json has no valid 'executable' entry: %s" % current_path)
    executable_path = Path(executable)
    if not executable_path.is_file():
        logger.warning("managed simulator executable not found: %s", executable_path)
        return None
    return str(executable_path)


def _managed_root(root: Optional[str]) -> Path:
    if root is None:
        return Path.home() / ".donkeycar" / "simulators"
    return Path(os.path.abspath(os.path.expanduser(str(root))))


class _FileLock:
    """Advisory cross-process lock file (fcntl on Unix, msvcrt on Windows)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(str(self.path), "a+b")
        if os.name == "nt":
            if os.fstat(self._fh.fileno()).st_size == 0:
                self._fh.write(b"\0")
                self._fh.flush()
            locking = getattr(msvcrt, "locking")
            lock_nonblocking = int(getattr(msvcrt, "LK_NBLCK"))
            while True:
                self._fh.seek(0)
                try:
                    locking(self._fh.fileno(), lock_nonblocking, 1)
                    break
                except OSError:
                    time.sleep(0.25)
        else:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if os.name == "nt":
                self._fh.seek(0)
                locking = getattr(msvcrt, "locking")
                locking(self._fh.fileno(), int(getattr(msvcrt, "LK_UNLCK")), 1)
            else:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


class SimulatorUpdater:
    """Download, verify, stage and apply updates for the Donkey Car simulator.

    :param repo: GitHub repository holding the releases (``owner/name``).
    :param root: managed directory; defaults to ``~/.donkeycar/simulators``.
    :param platform: ``sys.platform`` value selecting the asset to install.
    :param timeout: network timeout in seconds for API calls and downloads.
    :param progress: optional callback ``progress(downloaded_bytes, total_bytes)`` called
        while a download is streaming.
    :param cache_seconds: how long a GitHub release lookup stays valid; ``0`` disables cache reads.
    :param prerelease: when true, resolve updates from the newest pre-release instead of
        the latest stable release.
    """

    def __init__(
        self,
        repo: str = DEFAULT_SIMULATOR_UPDATE_REPO,
        root: Optional[str] = None,
        platform: str = sys.platform,
        timeout: float = 15.0,
        progress: Optional[Callable[[int, int], None]] = None,
        cache_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
        prerelease: bool = False,
    ):
        self.repo = repo
        self.root = _managed_root(root)
        self.platform = platform
        self.timeout = float(timeout)
        self.progress = progress
        self.cache_seconds = max(float(cache_seconds), 0.0)
        self.prerelease = bool(prerelease)

    def _channel_name(self) -> str:
        return "prerelease" if self.prerelease else "stable"

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    @property
    def downloads_dir(self) -> Path:
        return self.root / "downloads"

    @property
    def current_path(self) -> Path:
        return self.root / "current.json"

    @property
    def staged_path(self) -> Path:
        return self.root / "staged.json"

    @property
    def cache_path(self) -> Path:
        return self.root / "last_check.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "updates.lock"

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)

    def _asset_name(self) -> str:
        try:
            return _PLATFORM_ASSET_NAMES[self.platform]
        except KeyError:
            raise SimulatorUpdateError(
                "unsupported platform %r for simulator updates (expected darwin, win32 or linux)" % self.platform
            )

    def _read_cached_release(self) -> Optional[SimulatorRelease]:
        if self.cache_seconds <= 0:
            return None
        if not self.cache_path.exists():
            return None
        try:
            data = _load_json_file(self.cache_path)
        except SimulatorUpdateError:
            logger.warning("ignoring corrupted release cache: %s", self.cache_path)
            return None
        checked_at = data.get("checked_at")
        try:
            parsed = datetime.datetime.fromisoformat(checked_at) if isinstance(checked_at, str) else None
            if parsed is None or datetime.datetime.now(datetime.timezone.utc) - parsed > datetime.timedelta(
                seconds=self.cache_seconds
            ):
                return None
        except (TypeError, ValueError):
            return None
        if data.get("channel", "stable") != self._channel_name():
            return None
        release_data = data.get("release")
        if not isinstance(release_data, dict):
            return None
        try:
            release = _release_from_dict(release_data)
        except (KeyError, TypeError, ValueError):
            logger.warning("ignoring malformed release cache: %s", self.cache_path)
            return None
        if release.asset_name != self._asset_name():
            return None
        return release

    def _release_from_api(self, data: Any) -> SimulatorRelease:
        if not isinstance(data, dict):
            raise SimulatorUpdateError("unexpected GitHub API response format")
        tag = data.get("tag_name")
        if not isinstance(tag, str) or not tag.strip():
            raise SimulatorUpdateError("GitHub API release has no tag_name")
        asset_name = self._asset_name()
        selected = None
        for asset in data.get("assets") or []:
            if isinstance(asset, dict) and asset.get("name") == asset_name:
                selected = asset
                break
        if selected is None:
            raise SimulatorUpdateError("release %r has no %r asset for platform %r" % (tag, asset_name, self.platform))
        asset_url = selected.get("browser_download_url")
        if not isinstance(asset_url, str) or not asset_url.strip():
            raise SimulatorUpdateError("asset %r has no download URL" % asset_name)
        try:
            asset_size = int(selected.get("size") or 0)
        except (TypeError, ValueError):
            asset_size = 0
        digest = selected.get("digest")
        published_at = data.get("published_at")
        return SimulatorRelease(
            tag=tag,
            asset_name=asset_name,
            asset_url=asset_url,
            asset_size=asset_size,
            asset_sha256=digest if isinstance(digest, str) else "",
            published_at=published_at if isinstance(published_at, str) else "",
        )

    def _write_cache(self, release: SimulatorRelease) -> None:
        _atomic_write_json(
            self.cache_path, {"checked_at": _now_iso(), "channel": self._channel_name(), "release": _release_to_dict(release)}
        )

    def _fetch_api_json(self, url: str) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read() or b""
            except Exception:
                pass
            message = "GitHub API returned HTTP %s for %s" % (exc.code, url)
            if exc.code in (403, 429):
                if b"abuse" in body.lower():
                    message += ": GitHub abuse detection triggered, wait before retrying"
                elif exc.code == 429:
                    message += ": rate limit exceeded, wait before retrying"
                else:
                    message += ": access denied or rate limited"
                raise SimulatorRateLimitError(message) from exc
            raise SimulatorDownloadError(message) from exc
        except (urllib.error.URLError, HTTPException, socket.timeout, TimeoutError, OSError) as exc:
            raise SimulatorDownloadError("cannot reach GitHub API at %s: %s" % (url, exc)) from exc
        try:
            return json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise SimulatorUpdateError("GitHub API returned invalid JSON for %s: %s" % (url, exc)) from exc

    def fetch_latest(self, force: bool = False) -> SimulatorRelease:
        """Return the latest :class:`SimulatorRelease` for this platform.

        The result is cached in ``last_check.json`` for 24 hours; pass ``force=True`` to
        bypass the cache.  With ``prerelease=True`` the newest pre-release is resolved
        instead of the latest stable release.  The GitHub API is called exactly once per
        attempt (no retries); HTTP 403 / 429 / abuse-detection responses raise
        :class:`SimulatorRateLimitError`.
        """
        if not force:
            cached = self._read_cached_release()
            if cached is not None:
                return cached
        if not isinstance(self.repo, str) or "/" not in self.repo:
            raise SimulatorUpdateError("invalid GitHub repository: %r" % (self.repo,))
        self._asset_name()  # fail fast on unsupported platforms, before any network call
        if self.prerelease:
            return self._fetch_latest_prerelease()
        url = _API_LATEST_URL.format(repo=self.repo)
        data = self._fetch_api_json(url)
        release = self._release_from_api(data)
        self._write_cache(release)
        return release

    def _fetch_latest_prerelease(self) -> SimulatorRelease:
        url = _API_RELEASES_URL.format(repo=self.repo)
        data = self._fetch_api_json(url)
        if not isinstance(data, list):
            raise SimulatorUpdateError("unexpected GitHub API response format for %s" % url)
        for item in data:
            if isinstance(item, dict) and item.get("prerelease") is True:
                release = self._release_from_api(item)
                self._write_cache(release)
                return release
        raise SimulatorUpdateError("no pre-release found in the latest releases of %s" % self.repo)

    def get_installed(self) -> Optional[InstalledSimulator]:
        """Return the currently selected installed version, or ``None`` if none exists."""
        if not self.current_path.exists():
            return None
        data = _load_json_file(self.current_path)
        if not isinstance(data.get("tag"), str) or not isinstance(data.get("executable"), str):
            raise SimulatorUpdateError("current.json is missing 'tag' or 'executable': %s" % self.current_path)
        executable = Path(data["executable"])
        if not executable.is_file():
            logger.warning("recorded simulator executable does not exist: %s", executable)
            return None
        try:
            asset_size = int(data.get("asset_size") or 0)
        except (TypeError, ValueError):
            asset_size = 0
        return InstalledSimulator(
            tag=data["tag"],
            executable=str(executable),
            asset_name=data.get("asset_name") or "",
            asset_sha256=data.get("asset_sha256") or "",
            asset_size=asset_size,
            installed_at=data.get("installed_at") or "",
        )

    def check_for_update(self, force: bool = False) -> UpdateCheck:
        """Compare the installed version with the latest release (network unless cached)."""
        installed = self.get_installed()
        latest = self.fetch_latest(force=force)
        update_available = installed is None or installed.tag != latest.tag
        return UpdateCheck(installed=installed, latest=latest, update_available=update_available, checked_at=_now_iso())

    def _verify_file(self, path: Path, digest_hex: str, expected_size: int) -> bool:
        if expected_size > 0 and path.stat().st_size != expected_size:
            return False
        hasher = hashlib.sha256()
        try:
            with open(str(path), "rb") as fh:
                for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                    hasher.update(chunk)
        except OSError:
            return False
        return hasher.hexdigest().lower() == digest_hex.lower()

    def _download(self, release: SimulatorRelease, downloads_dir: Path) -> Path:
        """Download ``release`` with resume support, verify it, and return the final path.

        An interrupted transfer keeps its ``.part`` file for resumption.  A partial file
        that fails digest or size verification is discarded so the next attempt starts
        clean instead of resuming from corrupt data.
        """
        digest_hex = _parse_sha256_digest(release.asset_sha256, release.asset_name)
        downloads_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = sanitize_tag(release.tag)
        zip_name = "%s-%s" % (safe_tag, release.asset_name)
        part_path = downloads_dir / ("%s.part" % zip_name)
        final_path = downloads_dir / zip_name

        if final_path.exists():
            if self._verify_file(final_path, digest_hex, release.asset_size):
                return final_path
            _silent_unlink(final_path)

        resume_from = part_path.stat().st_size if part_path.exists() else 0
        if resume_from > 0 and release.asset_size > 0 and resume_from >= release.asset_size:
            _silent_unlink(part_path)
            resume_from = 0

        headers = {"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"}
        if resume_from > 0:
            headers["Range"] = "bytes=%d-" % resume_from
        request = urllib.request.Request(release.asset_url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read() or b""
            except Exception:
                pass
            message = "download of %r failed: HTTP %s" % (release.asset_name, exc.code)
            if exc.code in (403, 429):
                if b"abuse" in body.lower():
                    message += ": GitHub abuse detection triggered, wait before retrying"
                elif exc.code == 429:
                    message += ": rate limit exceeded, wait before retrying"
                else:
                    message += ": access denied or rate limited"
                raise SimulatorRateLimitError(message) from exc
            raise SimulatorDownloadError(message) from exc
        except (urllib.error.URLError, HTTPException, socket.timeout, TimeoutError, OSError) as exc:
            raise SimulatorDownloadError(
                "download interrupted for %r: %s (partial kept at %s)" % (release.asset_name, exc, part_path)
            ) from exc

        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode() or 200
        if status == 200:
            mode = "wb"
            downloaded = 0
            hasher = hashlib.sha256()
        elif status == 206 and resume_from > 0:
            mode = "ab"
            downloaded = resume_from
            hasher = hashlib.sha256()
            with open(str(part_path), "rb") as fh:
                for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                    hasher.update(chunk)
        else:
            response.close()
            raise SimulatorDownloadError("unexpected HTTP status %s while downloading %r" % (status, release.asset_name))

        try:
            with open(str(part_path), mode) as fh:
                try:
                    while True:
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        if self.progress is not None:
                            self.progress(downloaded, release.asset_size)
                except (OSError, HTTPException, socket.timeout, TimeoutError, urllib.error.URLError) as exc:
                    raise SimulatorDownloadError(
                        "download interrupted for %r: %s (partial kept at %s)" % (release.asset_name, exc, part_path)
                    ) from exc
                finally:
                    try:
                        response.close()
                    except Exception:
                        pass
        except OSError as exc:
            raise SimulatorDownloadError("cannot write partial download %s: %s" % (part_path, exc)) from exc

        if release.asset_size > 0 and downloaded != release.asset_size:
            _silent_unlink(part_path)
            raise SimulatorDownloadError(
                "size mismatch for %r: got %d bytes, expected %d (corrupt partial discarded)"
                % (release.asset_name, downloaded, release.asset_size)
            )
        if hasher.hexdigest().lower() != digest_hex.lower():
            _silent_unlink(part_path)
            raise SimulatorDownloadError(
                "sha256 mismatch for %r (corrupt partial discarded; the next attempt starts over)" % release.asset_name
            )
        os.replace(str(part_path), str(final_path))
        return final_path

    def _check_zip_entry(self, info: zipfile.ZipInfo) -> None:
        name = info.filename
        normalized = name.replace("\\", "/")
        if not normalized:
            raise SimulatorUpdateError("archive contains an entry with an empty name")
        if normalized.startswith("/") or (len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":"):
            raise SimulatorUpdateError("archive entry has an absolute path, refusing: %r" % name)
        if any(part == ".." for part in normalized.split("/")):
            raise SimulatorUpdateError("archive entry escapes the staging directory, refusing: %r" % name)
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise SimulatorUpdateError("archive entry is a symbolic link, refusing: %r" % name)

    def _extract(self, zip_path: Path, staging_dir: Path) -> None:
        try:
            archive = zipfile.ZipFile(str(zip_path))
        except (zipfile.BadZipFile, OSError) as exc:
            raise SimulatorUpdateError("downloaded archive is not a valid zip file: %s" % exc)
        with archive:
            for info in archive.infolist():
                self._check_zip_entry(info)
                target = staging_dir / info.filename
                if info.filename.endswith("/") or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFDIR:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with archive.open(info) as src, open(str(target), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
                    raise SimulatorUpdateError("failed to extract %r: %s" % (info.filename, exc))

    def _find_executable(self, directory: Path) -> Path:
        """Locate the simulator executable inside an extracted archive or version dir."""
        platform = self.platform
        if platform == "darwin":
            for app_dir in sorted(directory.rglob("donkey_sim.app")):
                macos_dir = app_dir / "Contents" / "MacOS"
                if not macos_dir.is_dir():
                    continue
                candidates = sorted(path for path in macos_dir.iterdir() if path.is_file())
                if candidates:
                    preferred = [path for path in candidates if path.name == "donkey_sim"]
                    return (preferred or candidates)[0]
            raise SimulatorUpdateError("no donkey_sim.app executable found in the downloaded archive")
        if platform == "win32":
            executables = sorted(path for path in directory.rglob("*.exe") if path.is_file())
            donkey_exes = [path for path in executables if path.name.lower().startswith("donkey_sim")]
            if donkey_exes:
                preferred = [path for path in donkey_exes if path.name.lower() == "donkey_sim.exe"]
                return (preferred or donkey_exes)[0]
            if executables:
                return executables[0]
            raise SimulatorUpdateError("no donkey_sim*.exe executable found in the downloaded archive")
        if platform in ("linux", "linux2"):
            candidates = sorted(path for path in directory.rglob("donkey_sim*") if path.is_file())
            if candidates:
                preferred = [path for path in candidates if path.name == "donkey_sim"]
                return (preferred or candidates)[0]
            raise SimulatorUpdateError("no donkey_sim executable found in the downloaded archive")
        raise SimulatorUpdateError("unsupported platform %r" % platform)

    def stage_update(self, release: Optional[SimulatorRelease] = None) -> Dict[str, Any]:
        """Download and extract a release into ``.staging-<safe-tag>-<uuid>``.

        When ``release`` is ``None`` the latest release is fetched (cached).  The staging
        record is written atomically to ``staged.json``; call :meth:`apply_staged_update`
        to make it the current version.  Returns the staged record as a dict.
        """
        if release is None:
            release = self.fetch_latest()
        if not isinstance(release, SimulatorRelease):
            raise SimulatorUpdateError("stage_update() expects a SimulatorRelease, got %r" % (release,))
        self._ensure_dirs()
        with _FileLock(self.lock_path):
            return self._stage_locked(release)

    def _stage_locked(self, release: SimulatorRelease) -> Dict[str, Any]:
        safe_tag = sanitize_tag(release.tag)
        staging_dir = self.root / (".staging-%s-%s" % (safe_tag, uuid.uuid4().hex))
        for leftover in self.root.glob(".staging-%s-*" % safe_tag):
            if leftover.is_dir():
                shutil.rmtree(str(leftover), ignore_errors=True)
        zip_path = self._download(release, self.downloads_dir)
        try:
            staging_dir.mkdir(parents=True)
            self._extract(zip_path, staging_dir)
            executable = self._find_executable(staging_dir)
            if os.name == "posix":
                os.chmod(
                    str(executable),
                    executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                )
            staged = {
                "tag": release.tag,
                "safe_tag": safe_tag,
                "executable": str(executable),
                "asset_name": release.asset_name,
                "asset_sha256": release.asset_sha256,
                "asset_size": release.asset_size,
                "staging_dir": str(staging_dir),
                "staged_at": _now_iso(),
            }
            _atomic_write_json(self.staged_path, staged)
            return dict(staged)
        except SimulatorUpdateError:
            shutil.rmtree(str(staging_dir), ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(str(staging_dir), ignore_errors=True)
            raise SimulatorUpdateError("failed to stage release %r: %s" % (release.tag, exc))

    def _load_staged(self) -> Dict[str, Any]:
        if not self.staged_path.exists():
            raise SimulatorUpdateError("no staged update found; call stage_update() first")
        data = _load_json_file(self.staged_path)
        for key in ("tag", "staging_dir", "executable"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise SimulatorUpdateError("staged.json is missing required field %r" % key)
        data.setdefault("asset_name", "")
        data.setdefault("asset_sha256", "")
        try:
            data["asset_size"] = int(data.get("asset_size") or 0)
        except (TypeError, ValueError):
            data["asset_size"] = 0
        return data

    def _write_version_metadata(self, target: Path, staged: Dict[str, Any], installed_at: str) -> None:
        metadata_path = target / _VERSION_METADATA_NAME
        if metadata_path.exists():
            return
        _atomic_write_json(
            metadata_path,
            {
                "tag": staged["tag"],
                "asset_name": staged["asset_name"],
                "asset_sha256": staged["asset_sha256"],
                "asset_size": staged["asset_size"],
                "installed_at": installed_at,
            },
        )

    def apply_staged_update(self, pid: Optional[int] = None, timeout: Optional[float] = None) -> InstalledSimulator:
        """Make the staged update the current version and return its record.

        If ``pid`` is given, the process is waited for (up to ``timeout`` seconds) before
        the staged directory is moved into place; when no pid is given the caller
        guarantees that no simulator process is still using the old version.  If the target
        version directory already exists it is reused and the staging directory is removed.
        """
        self._ensure_dirs()
        with _FileLock(self.lock_path):
            staged = self._load_staged()
            tag = staged["tag"]
            safe_tag = sanitize_tag(tag)
            staging_dir = Path(staged["staging_dir"])
            executable = Path(staged["executable"])
            if not staging_dir.is_dir():
                raise SimulatorUpdateError("staging directory is missing: %s" % staging_dir)
            try:
                executable.relative_to(staging_dir)
            except ValueError:
                raise SimulatorUpdateError("staged executable escapes the staging directory: %s" % executable)
            if pid is not None and not wait_for_pid(pid, timeout=timeout):
                raise SimulatorUpdateError("simulator process %s is still running; cannot apply the update" % pid)
            target = self.versions_dir / safe_tag
            installed_at = _now_iso()
            if target.is_dir():
                executable = self._find_executable(target)
                self._write_version_metadata(target, staged, installed_at)
                shutil.rmtree(str(staging_dir), ignore_errors=True)
            else:
                if not executable.is_file():
                    raise SimulatorUpdateError("staged executable is missing: %s" % executable)
                try:
                    os.replace(str(staging_dir), str(target))
                except OSError as exc:
                    raise SimulatorUpdateError("cannot move staged update into place: %s" % exc)
                executable = target / executable.relative_to(staging_dir)
                if not executable.is_file():
                    raise SimulatorUpdateError("staged executable is missing after install: %s" % executable)
                self._write_version_metadata(target, staged, installed_at)
            record = InstalledSimulator(
                tag=tag,
                executable=str(executable),
                asset_name=staged["asset_name"],
                asset_sha256=staged["asset_sha256"],
                asset_size=int(staged["asset_size"]),
                installed_at=installed_at,
            )
            _atomic_write_json(self.current_path, _installed_to_dict(record))
            _silent_unlink(self.staged_path)
            return record

    def install_update(
        self, release: Optional[SimulatorRelease] = None, pid: Optional[int] = None, timeout: Optional[float] = None
    ) -> InstalledSimulator:
        """Stage the latest release (or ``release``) and apply it immediately."""
        self.stage_update(release=release)
        return self.apply_staged_update(pid=pid, timeout=timeout)

    def use_version(self, tag: str) -> InstalledSimulator:
        """Select an already installed version directory and return its record."""
        self._ensure_dirs()
        with _FileLock(self.lock_path):
            safe_tag = sanitize_tag(tag)
            target = self.versions_dir / safe_tag
            if not target.is_dir():
                raise SimulatorUpdateError("simulator version %r is not installed (looked in %s)" % (tag, target))
            executable = self._find_executable(target)
            metadata = {}
            metadata_path = target / _VERSION_METADATA_NAME
            if metadata_path.exists():
                metadata = _load_json_file(metadata_path)
            try:
                asset_size = int(metadata.get("asset_size") or 0)
            except (TypeError, ValueError):
                asset_size = 0
            record = InstalledSimulator(
                tag=tag,
                executable=str(executable),
                asset_name=metadata.get("asset_name") or "",
                asset_sha256=metadata.get("asset_sha256") or "",
                asset_size=asset_size,
                installed_at=_now_iso(),
            )
            _atomic_write_json(self.current_path, _installed_to_dict(record))
            return record

    def clean(self, keep_versions: int = 2) -> List[str]:
        """Remove older version directories, keeping the ``keep_versions`` newest.

        The version referenced by ``current.json`` or ``staged.json`` is never removed, nor
        are directories whose names do not look like sanitized tags, nor orphaned staging
        directories referenced by a pending ``staged.json``.  Returns the removed tags.
        """
        self._ensure_dirs()
        with _FileLock(self.lock_path):
            if isinstance(keep_versions, bool) or not isinstance(keep_versions, int) or keep_versions < 1:
                raise SimulatorUpdateError("keep_versions must be a positive integer, got %r" % (keep_versions,))
            current_tag = self._read_optional_tag(self.current_path)
            staged_tag, staged_dir = self._read_optional_staged()
            protected = set()
            if current_tag is not None:
                protected.add(sanitize_tag(current_tag))
            if staged_tag is not None:
                protected.add(sanitize_tag(staged_tag))
            versions = [path for path in self.versions_dir.iterdir() if path.is_dir() and path.name == sanitize_tag(path.name)]
            versions.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            protected_count = len([path for path in versions if path.name in protected])
            quota = max(keep_versions - protected_count, 0)
            removed = []
            for path in versions:
                if path.name in protected:
                    continue
                if quota > 0:
                    quota -= 1
                    continue
                shutil.rmtree(str(path), ignore_errors=True)
                removed.append(path.name)
            for staging in self.root.glob(".staging-*"):
                if staging.is_dir() and str(staging) != staged_dir:
                    shutil.rmtree(str(staging), ignore_errors=True)
            return removed

    def _read_optional_tag(self, path: Path) -> Optional[str]:
        try:
            data = _load_json_file(path)
        except SimulatorUpdateError:
            return None
        tag = data.get("tag")
        return tag if isinstance(tag, str) and tag.strip() else None

    def _read_optional_staged(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            data = _load_json_file(self.staged_path)
        except SimulatorUpdateError:
            return None, None
        tag = data.get("tag")
        staging_dir = data.get("staging_dir")
        if not isinstance(tag, str) or not isinstance(staging_dir, str):
            return None, None
        return tag, staging_dir


__all__ = [
    "DEFAULT_SIMULATOR_UPDATE_REPO",
    "InstalledSimulator",
    "SimulatorDownloadError",
    "SimulatorRateLimitError",
    "SimulatorRelease",
    "SimulatorUpdateError",
    "SimulatorUpdater",
    "UpdateCheck",
    "resolve_managed_executable",
    "sanitize_tag",
    "wait_for_pid",
]
