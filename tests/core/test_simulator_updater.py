#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gym_donkeycar.core.simulator_updater`.

The module under test is loaded directly from its source file so that the test suite does
not require the runtime dependencies of the gym_donkeycar package (gymnasium is not
installed in every environment).  All tests are hermetic: an autouse fixture blocks every
network access, and tests that exercise downloads or the GitHub API install fakes.
"""

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import time
import urllib.error
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

_MODULE_FILE = pathlib.Path(__file__).resolve().parents[2] / "gym_donkeycar" / "core" / "simulator_updater.py"


def _load_simulator_updater():
    spec = importlib.util.spec_from_file_location("gym_donkeycar.core.simulator_updater", _MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


su = _load_simulator_updater()

DIGEST_HEX = "ab" * 32


def digest_of(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def make_release(**overrides):
    fields = {
        "tag": "v25.10.06",
        "asset_name": "DonkeySimMac.zip",
        "asset_url": "https://downloads.invalid/DonkeySimMac.zip",
        "asset_size": 0,
        "asset_sha256": "sha256:" + DIGEST_HEX,
        "published_at": "2025-10-06T00:00:00Z",
    }
    fields.update(overrides)
    return su.SimulatorRelease(**fields)


def github_release_payload(
    tag="v25.10.06",
    asset_name="DonkeySimMac.zip",
    digest=None,
    size=None,
    url="https://downloads.invalid/DonkeySimMac.zip",
):
    asset = {
        "name": asset_name,
        "browser_download_url": url,
        "digest": digest if digest is not None else "sha256:" + DIGEST_HEX,
    }
    if size is not None:
        asset["size"] = size
    return {"tag_name": tag, "published_at": "2025-10-06T00:00:00Z", "assets": [asset]}


def mac_zip_bytes(entry="DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim", content=b"#!/bin/sh\nexit 0\n"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, content)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status
        self._pos = 0

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if self._pos >= len(self._payload):
            return b""
        if size is None or size < 0:
            chunk = self._payload[self._pos :]
            self._pos = len(self._payload)
            return chunk
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LegacyFileUrlResponse(FakeResponse):
    """file:// responses expose status=None and report 200 through getcode()."""

    def __init__(self, payload):
        super().__init__(payload)
        self.status = None


class InterruptedResponse(FakeResponse):
    """Serve up to ``stop_after`` bytes, then raise as if the connection dropped."""

    def __init__(self, payload, stop_after):
        super().__init__(payload)
        self._stop_after = stop_after

    def read(self, size=-1):
        if self._pos >= self._stop_after:
            raise OSError("connection reset")
        step = 32 if size is None or size < 0 else min(size, 32)
        return super().read(step)


def http_error(url, code, message="error", body=b""):
    return urllib.error.HTTPError(url, code, message, {}, io.BytesIO(body) if body else None)


def install_urlopen(monkeypatch, handler):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        result = handler(request, timeout)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(su.urllib.request, "urlopen", fake_urlopen)
    return calls


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(request, timeout=None):
        raise AssertionError("test attempted network access to %s" % request.full_url)

    monkeypatch.setattr(su.urllib.request, "urlopen", blocked)


def write_zip(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def write_zip_with_symlink(path, name, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), "w") as archive:
        info = zipfile.ZipInfo(name)
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)
    return path


def write_current_json(root, tag, executable):
    current = {
        "tag": tag,
        "executable": executable,
        "asset_name": "DonkeySimMac.zip",
        "asset_sha256": "sha256:" + DIGEST_HEX,
        "asset_size": 10,
        "installed_at": "2025-10-06T00:00:00+00:00",
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(str(root / "current.json"), "w") as fh:
        json.dump(current, fh)
    return current


def make_installed_version(root, tag, platform="darwin"):
    safe_tag = su.sanitize_tag(tag)
    version_dir = root / "versions" / safe_tag
    if platform == "darwin":
        executable = version_dir / "DonkeySimMac" / "donkey_sim.app" / "Contents" / "MacOS" / "donkey_sim"
    elif platform == "linux":
        executable = version_dir / "DonkeySimLinux" / "donkey_sim"
    else:
        executable = version_dir / "DonkeySimWin" / "donkey_sim.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"#!/bin/sh\n")
    metadata = version_dir / "version.json"
    metadata.write_text(
        json.dumps(
            {
                "tag": tag,
                "asset_name": "asset.zip",
                "asset_sha256": "sha256:" + DIGEST_HEX,
                "asset_size": 10,
                "installed_at": "2025-10-06T00:00:00+00:00",
            }
        )
    )
    return version_dir, executable


@contextmanager
def spawn_sleeper(seconds):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(%s)" % seconds])
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


# ---------------------------------------------------------------------------
# constants, construction, platform mapping
# ---------------------------------------------------------------------------


def test_default_repo_constant():
    assert su.DEFAULT_SIMULATOR_UPDATE_REPO == "tawnkramer/gym-donkeycar"


def test_default_root_is_home_donkeycar_simulators():
    updater = su.SimulatorUpdater()
    assert updater.root == pathlib.Path.home() / ".donkeycar" / "simulators"


def test_custom_root(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    assert updater.root == tmp_path
    assert updater.root.is_absolute()


def test_platform_asset_names():
    assert su.SimulatorUpdater(platform="darwin")._asset_name() == "DonkeySimMac.zip"
    assert su.SimulatorUpdater(platform="win32")._asset_name() == "DonkeySimWin.zip"
    assert su.SimulatorUpdater(platform="linux")._asset_name() == "DonkeySimLinux.zip"
    assert su.SimulatorUpdater(platform="linux2")._asset_name() == "DonkeySimLinux.zip"


def test_unsupported_platform_raises(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="freebsd")
    with pytest.raises(su.SimulatorUpdateError, match="unsupported platform"):
        updater.fetch_latest()


def test_invalid_repo_raises(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path), repo="not-a-repo")
    with pytest.raises(su.SimulatorUpdateError, match="repository"):
        updater.fetch_latest()


# ---------------------------------------------------------------------------
# fetch_latest: release parsing, asset selection, caching, errors
# ---------------------------------------------------------------------------


def test_fetch_latest_parses_release_and_selects_platform_asset(tmp_path, monkeypatch):
    payload = github_release_payload(size=123, digest="sha256:" + DIGEST_HEX)
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))

    updater = su.SimulatorUpdater(root=str(tmp_path), platform="darwin")
    release = updater.fetch_latest()

    assert release.tag == "v25.10.06"
    assert release.asset_name == "DonkeySimMac.zip"
    assert release.asset_url == "https://downloads.invalid/DonkeySimMac.zip"
    assert release.asset_size == 123
    assert release.asset_sha256 == "sha256:" + DIGEST_HEX
    assert release.published_at == "2025-10-06T00:00:00Z"
    assert calls[0][0].full_url == "https://api.github.com/repos/tawnkramer/gym-donkeycar/releases/latest"
    assert (
        calls[0][0].get_header("User-agent")
        == "gym-donkeycar-simulator-updater/1.0 (+https://github.com/tawnkramer/gym-donkeycar)"
    )
    assert json.loads((tmp_path / "last_check.json").read_text())["release"]["tag"] == "v25.10.06"


def test_fetch_latest_caches_within_24_hours(tmp_path, monkeypatch):
    payload = github_release_payload()
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    first = updater.fetch_latest()
    second = updater.fetch_latest()

    assert first == second
    assert len(calls) == 1


def test_fetch_latest_force_bypasses_cache(tmp_path, monkeypatch):
    payload = github_release_payload()
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    updater.fetch_latest()
    updater.fetch_latest(force=True)

    assert len(calls) == 2


def test_fetch_latest_honors_custom_cache_ttl(tmp_path, monkeypatch):
    payload = github_release_payload()
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path), cache_seconds=3600)
    cache_path = tmp_path / "last_check.json"

    updater.fetch_latest()
    cache = json.loads(cache_path.read_text())
    cache["checked_at"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    cache_path.write_text(json.dumps(cache))
    updater.fetch_latest()

    assert len(calls) == 1

    cache["checked_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    cache_path.write_text(json.dumps(cache))
    updater.fetch_latest()

    assert len(calls) == 2


def test_fetch_latest_stale_cache_refetches(tmp_path, monkeypatch):
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "last_check.json").write_text(
        json.dumps(
            {
                "checked_at": stale,
                "release": su.SimulatorRelease(
                    tag="vold",
                    asset_name="DonkeySimMac.zip",
                    asset_url="https://x.invalid",
                    asset_size=1,
                    asset_sha256="sha256:" + DIGEST_HEX,
                    published_at="2020-01-01T00:00:00Z",
                ).__dict__,
            }
        )
    )
    payload = github_release_payload()
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    assert updater.fetch_latest().tag == "v25.10.06"
    assert len(calls) == 1


def test_fetch_latest_corrupted_cache_refetches(tmp_path, monkeypatch):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "last_check.json").write_text("{not json")
    payload = github_release_payload()
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    assert updater.fetch_latest().tag == "v25.10.06"
    assert len(calls) == 1
    assert json.loads((tmp_path / "last_check.json").read_text())["checked_at"]


def test_fetch_latest_rate_limit_raises(tmp_path, monkeypatch):
    def handler(request, timeout):
        return http_error(request.full_url, 403, "Forbidden", b"You have triggered an abuse detection mechanism")

    install_urlopen(monkeypatch, handler)
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorRateLimitError, match="abuse"):
        updater.fetch_latest()


def test_fetch_latest_http_429_raises(tmp_path, monkeypatch):
    install_urlopen(monkeypatch, lambda request, timeout: http_error(request.full_url, 429, "Too Many Requests"))
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorRateLimitError, match="rate limit"):
        updater.fetch_latest()


def test_fetch_latest_http_403_raises(tmp_path, monkeypatch):
    install_urlopen(monkeypatch, lambda request, timeout: http_error(request.full_url, 403, "Forbidden"))
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorRateLimitError, match="403"):
        updater.fetch_latest()


def test_fetch_latest_http_404_raises(tmp_path, monkeypatch):
    install_urlopen(monkeypatch, lambda request, timeout: http_error(request.full_url, 404, "Not Found"))
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorDownloadError, match="404"):
        updater.fetch_latest()


def test_fetch_latest_network_error_raises(tmp_path, monkeypatch):
    install_urlopen(monkeypatch, lambda request, timeout: urllib.error.URLError("boom"))
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorDownloadError, match="boom"):
        updater.fetch_latest()


def test_fetch_latest_invalid_json_raises(tmp_path, monkeypatch):
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(b"not json at all"))
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorUpdateError, match="invalid JSON"):
        updater.fetch_latest()


def test_fetch_latest_missing_platform_asset_raises(tmp_path, monkeypatch):
    payload = {"tag_name": "v25.10.06", "assets": [{"name": "DonkeySimWin.zip", "browser_download_url": "https://x.invalid"}]}
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="darwin")
    with pytest.raises(su.SimulatorUpdateError, match="DonkeySimMac.zip"):
        updater.fetch_latest()


# ---------------------------------------------------------------------------
# get_installed / check_for_update
# ---------------------------------------------------------------------------


def test_get_installed_returns_none_when_absent(tmp_path):
    assert su.SimulatorUpdater(root=str(tmp_path)).get_installed() is None


def test_get_installed_corrupted_current_raises(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "current.json").write_text("{broken")
    with pytest.raises(su.SimulatorUpdateError, match="current.json"):
        su.SimulatorUpdater(root=str(tmp_path)).get_installed()


def test_get_installed_returns_record(tmp_path):
    root = tmp_path
    executable = root / "versions" / "v1" / "donkey_sim"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"x")
    write_current_json(root, "v1", str(executable))

    installed = su.SimulatorUpdater(root=str(root)).get_installed()
    assert installed.tag == "v1"
    assert installed.executable == str(executable)
    assert installed.asset_size == 10


def test_get_installed_none_when_executable_missing(tmp_path):
    root = tmp_path
    write_current_json(root, "v1", str(root / "versions" / "v1" / "gone"))
    assert su.SimulatorUpdater(root=str(root)).get_installed() is None


def test_check_for_update_nothing_installed(tmp_path, monkeypatch):
    payload = github_release_payload()
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    result = updater.check_for_update()

    assert result.installed is None
    assert result.latest.tag == "v25.10.06"
    assert result.update_available is True
    datetime.fromisoformat(result.checked_at)


def test_check_for_update_same_tag_not_available(tmp_path, monkeypatch):
    payload = github_release_payload()
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    root = tmp_path
    version_dir, executable = make_installed_version(root, "v25.10.06")
    write_current_json(root, "v25.10.06", str(executable))

    result = su.SimulatorUpdater(root=str(root)).check_for_update()

    assert result.update_available is False
    assert result.installed.tag == "v25.10.06"


def test_check_for_update_newer_tag_available(tmp_path, monkeypatch):
    payload = github_release_payload(tag="v26.0.0")
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    root = tmp_path
    version_dir, executable = make_installed_version(root, "v25.10.06")
    write_current_json(root, "v25.10.06", str(executable))

    result = su.SimulatorUpdater(root=str(root)).check_for_update()

    assert result.update_available is True
    assert result.latest.tag == "v26.0.0"


# ---------------------------------------------------------------------------
# staging: digest requirements, download, resume, verification
# ---------------------------------------------------------------------------


def test_stage_refuses_missing_digest(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    for bad in ("", None):
        release = make_release(asset_sha256=bad)
        with pytest.raises(su.SimulatorDownloadError, match="digest"):
            updater.stage_update(release)


def test_stage_refuses_malformed_digest(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    for bad in ("sha256:xyz", DIGEST_HEX, "sha256:" + "ab" * 32 + ":", "md5:" + "ab" * 32):
        release = make_release(asset_sha256=bad)
        with pytest.raises(su.SimulatorDownloadError, match="digest"):
            updater.stage_update(release)


def test_stage_accepts_response_without_status_attribute(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: LegacyFileUrlResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="darwin")

    staged = updater.stage_update(release)

    assert staged["tag"] == "v25.10.06"
    assert pathlib.Path(staged["executable"]).is_file()


def test_stage_downloads_extracts_and_writes_staged_json(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="darwin")

    staged = updater.stage_update(release)

    assert len(calls) == 1
    assert staged["tag"] == "v25.10.06"
    assert staged["safe_tag"] == "v25.10.06"
    assert staged["executable"].endswith("donkey_sim.app/Contents/MacOS/donkey_sim")
    assert staged["asset_name"] == "DonkeySimMac.zip"
    assert staged["asset_sha256"] == digest_of(payload)
    assert staged["staging_dir"].startswith(str(tmp_path))
    assert pathlib.Path(staged["executable"]).is_file()
    assert os.access(staged["executable"], os.X_OK)
    assert json.loads((tmp_path / "staged.json").read_text())["tag"] == "v25.10.06"
    assert (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip").is_file()
    assert (tmp_path / "updates.lock").exists()


def test_stage_resumes_interrupted_download(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    state = {"calls": 0}

    def handler(request, timeout):
        state["calls"] += 1
        if state["calls"] == 1:
            assert request.get_header("Range") is None
            return InterruptedResponse(payload, len(payload) // 2)
        part_path = tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip.part"
        part_size = part_path.stat().st_size
        assert request.get_header("Range") == "bytes=%d-" % part_size
        return FakeResponse(payload[part_size:], status=206)

    install_urlopen(monkeypatch, handler)
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorDownloadError, match="partial kept"):
        updater.stage_update(release)
    assert (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip.part").exists()

    staged = updater.stage_update(release)
    assert state["calls"] == 2
    assert staged["tag"] == "v25.10.06"
    downloaded = (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip").read_bytes()
    assert downloaded == payload
    assert not (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip.part").exists()


def test_stage_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    state = {"calls": 0}

    def handler(request, timeout):
        state["calls"] += 1
        if state["calls"] == 1:
            return InterruptedResponse(payload, len(payload) // 3)
        return FakeResponse(payload, status=200)

    install_urlopen(monkeypatch, handler)
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorDownloadError):
        updater.stage_update(release)
    updater.stage_update(release)

    downloaded = (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip").read_bytes()
    assert downloaded == payload


def test_stage_digest_mismatch_discards_part_and_raises(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    tampered = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(tampered))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorDownloadError, match="sha256 mismatch"):
        updater.stage_update(release)
    assert not (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip.part").exists()
    assert not (tmp_path / "downloads" / "v25.10.06-DonkeySimMac.zip").exists()
    assert not (tmp_path / "staged.json").exists()


def test_stage_size_mismatch_raises(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload) + 7, asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorDownloadError, match="size mismatch"):
        updater.stage_update(release)


def test_stage_progress_callback_reports_bytes(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    reports = []
    updater = su.SimulatorUpdater(root=str(tmp_path), progress=lambda done, total: reports.append((done, total)))

    updater.stage_update(release)

    assert reports
    assert reports[-1] == (len(payload), len(payload))
    assert all(b >= a for (a, _), (b, _) in zip(reports, reports[1:]))


def test_stage_reuses_verified_existing_download(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "v25.10.06-DonkeySimMac.zip").write_bytes(payload)
    calls = install_urlopen(monkeypatch, lambda request, timeout: AssertionError("network accessed"))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    updater.stage_update(release)

    assert calls == []


def test_stage_redownloads_corrupt_existing_download(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "v25.10.06-DonkeySimMac.zip").write_bytes(b"corrupted bytes")
    calls = install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    updater.stage_update(release)

    assert len(calls) == 1
    downloaded_hash = hashlib.sha256((downloads / "v25.10.06-DonkeySimMac.zip").read_bytes()).hexdigest()
    assert downloaded_hash == hashlib.sha256(payload).hexdigest()


def test_stage_rate_limit_during_download_raises(tmp_path, monkeypatch):
    release = make_release()
    install_urlopen(monkeypatch, lambda request, timeout: http_error(request.full_url, 403, "Forbidden"))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorRateLimitError):
        updater.stage_update(release)
    assert not (tmp_path / "staged.json").exists()


# ---------------------------------------------------------------------------
# safe extraction: zip-slip, symlinks, executable discovery
# ---------------------------------------------------------------------------


def test_extract_rejects_absolute_paths(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    for name in ("/absolute/path.txt", "//server/share.txt", "C:/evil.txt", "C:\\evil.txt"):
        zip_path = write_zip(tmp_path / "abs.zip", {name: b"x"})
        staging = tmp_path / "staging"
        staging.mkdir(exist_ok=True)
        with pytest.raises(su.SimulatorUpdateError, match="absolute"):
            updater._extract(zip_path, staging)
    assert not (tmp_path / "path.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_parent_traversal(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    for name in ("../evil.txt", "a/../../evil.txt", "..\\evil.txt", "a/..\\evil.txt"):
        zip_path = write_zip(tmp_path / "trav.zip", {name: b"x"})
        staging = tmp_path / "staging"
        staging.mkdir(exist_ok=True)
        with pytest.raises(su.SimulatorUpdateError, match="escapes"):
            updater._extract(zip_path, staging)
    assert not (tmp_path / "evil.txt").exists()


def test_extract_rejects_symlink_entries(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    zip_path = write_zip_with_symlink(tmp_path / "link.zip", "DonkeySimMac/link.txt", "/etc/passwd")
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(su.SimulatorUpdateError, match="symbolic link"):
        updater._extract(zip_path, staging)
    assert not (tmp_path / "staging" / "DonkeySimMac" / "link.txt").exists()


def test_stage_hostile_tag_cannot_escape_root(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release = make_release(tag="../../evil", asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    staged = updater.stage_update(release)

    root = tmp_path.resolve()
    staging_dir = pathlib.Path(staged["staging_dir"]).resolve()
    assert staging_dir.parent == root
    assert not staged["safe_tag"].startswith(".")
    assert staged["safe_tag"] not in (".", "..")
    assert "/" not in staged["safe_tag"]
    assert "\\" not in staged["safe_tag"]


def test_stage_linux_executable_discovery(tmp_path, monkeypatch):
    files = {"DonkeySimLinux/readme.txt": b"hi", "DonkeySimLinux/bin/donkey_sim": b"#!/bin/sh\n"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    payload = buffer.getvalue()
    release = make_release(asset_name="DonkeySimLinux.zip", asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="linux")

    staged = updater.stage_update(release)

    assert staged["executable"].endswith("DonkeySimLinux/bin/donkey_sim")
    assert os.access(staged["executable"], os.X_OK)


def test_stage_windows_prefers_donkey_sim_exe(tmp_path, monkeypatch):
    files = {"DonkeySimWin/other.exe": b"x", "DonkeySimWin/donkey_sim.exe": b"x"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    payload = buffer.getvalue()
    release = make_release(asset_name="DonkeySimWin.zip", asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="win32")

    staged = updater.stage_update(release)

    assert staged["executable"].endswith("donkey_sim.exe")


def test_stage_windows_falls_back_to_any_donkey_sim_exe(tmp_path, monkeypatch):
    files = {"DonkeySimWin/donkey_sim_launcher.exe": b"x", "DonkeySimWin/other.exe": b"x"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    payload = buffer.getvalue()
    release = make_release(asset_name="DonkeySimWin.zip", asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform="win32")

    staged = updater.stage_update(release)

    assert staged["executable"].endswith("donkey_sim_launcher.exe")


def test_stage_no_executable_raises_and_cleans_staging(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "no simulator here")
    payload = buffer.getvalue()
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorUpdateError, match="executable"):
        updater.stage_update(release)
    assert not list(tmp_path.glob(".staging-*"))
    assert not (tmp_path / "staged.json").exists()


def test_stage_invalid_zip_raises(tmp_path, monkeypatch):
    payload = b"this is not a zip archive"
    release = make_release(asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorUpdateError, match="not a valid zip"):
        updater.stage_update(release)
    assert not (tmp_path / "staged.json").exists()


# ---------------------------------------------------------------------------
# apply: rename, current.json, pid wait, reuse, corruption
# ---------------------------------------------------------------------------


def _stage_via_fake(tmp_path, monkeypatch, platform="darwin", tag="v25.10.06"):
    if platform == "darwin":
        payload = mac_zip_bytes()
        asset_name = "DonkeySimMac.zip"
    else:
        files = {"DonkeySimLinux/donkey_sim": b"#!/bin/sh\n"}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        payload = buffer.getvalue()
        asset_name = "DonkeySimLinux.zip"
    release = make_release(tag=tag, asset_name=asset_name, asset_size=len(payload), asset_sha256=digest_of(payload))
    install_urlopen(monkeypatch, lambda request, timeout: FakeResponse(payload))
    updater = su.SimulatorUpdater(root=str(tmp_path), platform=platform)
    updater.stage_update(release)
    return updater


def test_apply_moves_staging_and_writes_current(tmp_path, monkeypatch):
    updater = _stage_via_fake(tmp_path, monkeypatch)

    record = updater.apply_staged_update()

    version_dir = tmp_path / "versions" / "v25.10.06"
    assert record.tag == "v25.10.06"
    assert pathlib.Path(record.executable).is_file()
    assert record.executable.startswith(str(version_dir))
    assert record.executable.endswith("donkey_sim.app/Contents/MacOS/donkey_sim")
    assert (version_dir / "version.json").is_file()
    assert not (tmp_path / "staged.json").exists()
    assert not list(tmp_path.glob(".staging-*"))
    current = json.loads((tmp_path / "current.json").read_text())
    assert current["tag"] == "v25.10.06"
    assert current["executable"] == record.executable
    installed = updater.get_installed()
    assert installed.tag == "v25.10.06"
    assert installed.executable == record.executable


def test_apply_reuses_existing_version_dir(tmp_path, monkeypatch):
    updater = _stage_via_fake(tmp_path, monkeypatch)
    version_dir, existing_executable = make_installed_version(tmp_path, "v25.10.06")
    (version_dir / "PREEXISTING").write_text("keep me")
    version_dir.joinpath("version.json").unlink()

    record = updater.apply_staged_update()

    assert record.executable == str(existing_executable)
    assert (version_dir / "PREEXISTING").read_text() == "keep me"
    assert (version_dir / "version.json").is_file()
    assert not (tmp_path / "staged.json").exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_apply_without_staged_raises(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorUpdateError, match="no staged update"):
        updater.apply_staged_update()


def test_apply_corrupted_staged_raises(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "staged.json").write_text("{broken")
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorUpdateError, match="staged.json"):
        updater.apply_staged_update()


def test_apply_waits_for_pid(tmp_path, monkeypatch):
    updater = _stage_via_fake(tmp_path, monkeypatch)
    with spawn_sleeper(30) as sleeper:
        assert not su.wait_for_pid(sleeper.pid, timeout=0.2)
        with pytest.raises(su.SimulatorUpdateError, match="still running"):
            updater.apply_staged_update(pid=sleeper.pid, timeout=0.2)
        assert not (tmp_path / "current.json").exists()
        sleeper.terminate()
        sleeper.wait()
        record = updater.apply_staged_update(pid=sleeper.pid, timeout=5)
    assert record.tag == "v25.10.06"
    assert (tmp_path / "current.json").exists()


def test_apply_ignores_pid_none(tmp_path, monkeypatch):
    updater = _stage_via_fake(tmp_path, monkeypatch)
    record = updater.apply_staged_update(pid=None)
    assert record.tag == "v25.10.06"


def test_atomic_writes_leave_no_temp_files(tmp_path, monkeypatch):
    updater = _stage_via_fake(tmp_path, monkeypatch)
    updater.apply_staged_update()
    assert list(tmp_path.rglob("*.tmp-*")) == []


# ---------------------------------------------------------------------------
# install_update end to end
# ---------------------------------------------------------------------------


def test_install_update_end_to_end(tmp_path, monkeypatch):
    payload = mac_zip_bytes()
    release_payload = github_release_payload(size=len(payload), digest=digest_of(payload))
    calls = install_urlopen(
        monkeypatch,
        lambda request, timeout: (
            FakeResponse(json.dumps(release_payload).encode())
            if "api.github.com" in request.full_url
            else FakeResponse(payload)
        ),
    )
    updater = su.SimulatorUpdater(root=str(tmp_path))

    record = updater.install_update()

    assert record.tag == "v25.10.06"
    assert pathlib.Path(record.executable).is_file()
    assert updater.get_installed().tag == "v25.10.06"
    assert updater.check_for_update().update_available is False
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# use_version
# ---------------------------------------------------------------------------


def test_use_version_switches_current(tmp_path):
    root = tmp_path
    _, old_executable = make_installed_version(root, "v1")
    _, new_executable = make_installed_version(root, "v2")
    write_current_json(root, "v2", str(new_executable))
    updater = su.SimulatorUpdater(root=str(root))

    record = updater.use_version("v1")

    assert record.tag == "v1"
    assert record.executable == str(old_executable)
    assert updater.get_installed().tag == "v1"
    assert json.loads((root / "current.json").read_text())["executable"] == str(old_executable)


def test_use_version_missing_raises(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    with pytest.raises(su.SimulatorUpdateError, match="not installed"):
        updater.use_version("v9.9.9")


def test_use_version_rejects_traversal_tag(tmp_path):
    root = tmp_path
    make_installed_version(root, "v1")
    updater = su.SimulatorUpdater(root=str(root))
    with pytest.raises(su.SimulatorUpdateError):
        updater.use_version("../../etc")
    assert not (root / "versions" / "etc").exists()


# ---------------------------------------------------------------------------
# sanitize_tag
# ---------------------------------------------------------------------------


def test_sanitize_tag():
    assert su.sanitize_tag("v1.0") == "v1.0"
    assert su.sanitize_tag("release_2-rc.1") == "release_2-rc.1"
    assert su.sanitize_tag(".") == "unknown"
    assert su.sanitize_tag("..") == "unknown"
    for tag in ("", None, 123, "   "):
        with pytest.raises(su.SimulatorUpdateError, match="invalid"):
            su.sanitize_tag(tag)
    for hostile in ("../../etc/passwd", "..\\..\\windows", "/abs", "v1.0/"):
        safe = su.sanitize_tag(hostile)
        assert safe not in (".", "..")
        assert "/" not in safe and "\\" not in safe
        assert not safe.startswith(".")
    assert len(su.sanitize_tag("a" * 500)) <= 120


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def _make_version_dirs(root, tags):
    for tag in tags:
        make_installed_version(root, tag)
    return [root / "versions" / su.sanitize_tag(tag) for tag in tags]


def test_clean_keeps_newest_and_protects_current(tmp_path):
    root = tmp_path
    oldest, middle, newest = _make_version_dirs(root, ["v1", "v2", "v3"])
    now = time.time()
    os.utime(str(oldest), (now - 300, now - 300))
    os.utime(str(middle), (now - 200, now - 200))
    os.utime(str(newest), (now - 100, now - 100))
    write_current_json(root, "v1", str(oldest / "DonkeySimMac" / "donkey_sim.app" / "Contents" / "MacOS" / "donkey_sim"))

    removed = su.SimulatorUpdater(root=str(root)).clean(keep_versions=2)

    assert removed == ["v2"]
    assert oldest.is_dir() and newest.is_dir()
    assert not middle.exists()


def test_clean_protects_staged_version(tmp_path):
    root = tmp_path
    old, new = _make_version_dirs(root, ["v1", "v2"])
    now = time.time()
    os.utime(str(old), (now - 200, now - 200))
    os.utime(str(new), (now - 100, now - 100))
    (root / "staged.json").write_text(
        json.dumps({"tag": "v1", "staging_dir": str(root / "staging-x"), "executable": str(root / "staging-x" / "s")})
    )

    removed = su.SimulatorUpdater(root=str(root)).clean(keep_versions=1)

    assert removed == ["v2"]
    assert old.is_dir()
    assert not new.exists()


def test_clean_ignores_non_version_dirs(tmp_path):
    root = tmp_path
    version_dir, _ = make_installed_version(root, "v1")
    weird = root / "versions" / "weird name!!"
    weird.mkdir(parents=True)
    (weird / "keep.txt").write_text("x")
    now = time.time()
    os.utime(str(version_dir), (now - 100, now - 100))
    os.utime(str(weird), (now - 50, now - 50))

    removed = su.SimulatorUpdater(root=str(root)).clean(keep_versions=1)

    assert removed == []
    assert (weird / "keep.txt").is_file()


def test_clean_removes_orphan_staging_dirs(tmp_path):
    root = tmp_path
    orphan = root / ".staging-v9-abcdef"
    orphan.mkdir(parents=True)
    (orphan / "file").write_text("x")

    su.SimulatorUpdater(root=str(root)).clean(keep_versions=2)

    assert not orphan.exists()


def test_clean_invalid_keep_raises(tmp_path):
    updater = su.SimulatorUpdater(root=str(tmp_path))
    for bad in (0, -1, True, "2"):
        with pytest.raises(su.SimulatorUpdateError, match="keep_versions"):
            updater.clean(keep_versions=bad)


def test_clean_tolerates_corrupted_current(tmp_path):
    root = tmp_path
    old, new = _make_version_dirs(root, ["v1", "v2"])
    now = time.time()
    os.utime(str(old), (now - 200, now - 200))
    os.utime(str(new), (now - 100, now - 100))
    (root / "current.json").write_text("{broken")

    removed = su.SimulatorUpdater(root=str(root)).clean(keep_versions=1)

    assert removed == ["v1"]
    assert new.is_dir()


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------


def test_lock_serializes_concurrent_clean(tmp_path):
    fcntl = pytest.importorskip("fcntl", reason="POSIX flock required")
    root = tmp_path
    old, new = _make_version_dirs(root, ["v1", "v2"])
    now = time.time()
    os.utime(str(old), (now - 200, now - 200))
    os.utime(str(new), (now - 100, now - 100))

    script = (
        "import importlib.util, pathlib\n"
        "p = pathlib.Path(%r)\n" % str(_MODULE_FILE)
        + "spec = importlib.util.spec_from_file_location('gym_donkeycar.core.simulator_updater', p)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "u = m.SimulatorUpdater(root=%r)\n" % str(root) + "print(u.clean(1))\n"
    )
    lock_file = open(str(root / "updates.lock"), "a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.0)
        assert process.poll() is None, "clean() must block while another process holds updates.lock"
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=30)
    finally:
        lock_file.close()
        if "process" in locals() and process.poll() is None:
            process.terminate()
    assert process.returncode == 0, stderr.decode()
    assert "v1" in stdout.decode()


def test_lock_released_after_failed_stage(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl", reason="POSIX flock required")
    release = make_release()
    install_urlopen(monkeypatch, lambda request, timeout: urllib.error.URLError("boom"))
    updater = su.SimulatorUpdater(root=str(tmp_path))

    with pytest.raises(su.SimulatorDownloadError):
        updater.stage_update(release)

    lock_file = open(str(tmp_path / "updates.lock"), "a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


# ---------------------------------------------------------------------------
# wait_for_pid
# ---------------------------------------------------------------------------


def test_wait_for_pid_exited_process():
    with spawn_sleeper(0.3) as sleeper:
        sleeper.wait()
        assert su.wait_for_pid(sleeper.pid, timeout=5) is True


def test_wait_for_pid_live_process_times_out():
    with spawn_sleeper(30) as sleeper:
        started = time.monotonic()
        assert su.wait_for_pid(sleeper.pid, timeout=0.3) is False
        assert time.monotonic() - started < 5


def test_wait_for_pid_live_process_without_timeout():
    with spawn_sleeper(0.5) as sleeper:
        assert su.wait_for_pid(sleeper.pid) is True


def test_wait_for_pid_invalid_pids():
    assert su.wait_for_pid(0) is True
    assert su.wait_for_pid(-1) is True
    assert su.wait_for_pid(None) is True


# ---------------------------------------------------------------------------
# resolve_managed_executable
# ---------------------------------------------------------------------------


def test_resolve_managed_executable_missing_root(tmp_path):
    assert su.resolve_managed_executable(root=str(tmp_path / "nope")) is None


def test_resolve_managed_executable_valid(tmp_path):
    executable = tmp_path / "sim" / "donkey_sim"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"x")
    write_current_json(tmp_path, "v1", str(executable))

    assert su.resolve_managed_executable(root=str(tmp_path)) == str(executable)


def test_resolve_managed_executable_corrupted_raises(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "current.json").write_text("{broken")
    with pytest.raises(su.SimulatorUpdateError, match="current.json"):
        su.resolve_managed_executable(root=str(tmp_path))


def test_resolve_managed_executable_gone_file(tmp_path):
    write_current_json(tmp_path, "v1", str(tmp_path / "gone" / "donkey_sim"))
    assert su.resolve_managed_executable(root=str(tmp_path)) is None
