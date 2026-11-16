#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the `donkey-sim-update` command line interface."""

import os
import sys
import types
from types import SimpleNamespace

import pytest

# The CLI binds names from the core updater module at import time. Register a
# stub before importing it so these tests exercise only the CLI's own logic;
# each test monkeypatches the CLI's bindings with fakes, so nothing touches
# the network or any path outside tmp_path.
_core_updater_stub = types.ModuleType("gym_donkeycar.core.simulator_updater")
_core_updater_stub.DEFAULT_SIMULATOR_UPDATE_REPO = "tawnkramer/gym-donkeycar"
_core_updater_stub.SimulatorUpdater = object
_core_updater_stub.resolve_managed_executable = None
_core_updater_stub.wait_for_pid = None
sys.modules["gym_donkeycar.core.simulator_updater"] = _core_updater_stub

import gym_donkeycar.simulator_update as simulator_update


class FakeUpdater:
    instances = []

    def __init__(self, repo=None, root=None, platform=None, timeout=None, progress=None):
        FakeUpdater.instances.append(self)
        self.repo = repo
        self.root = root
        self.platform = platform
        self.timeout = timeout
        self.progress = progress
        self.installed = "25.10.06"
        self.latest_info = SimpleNamespace(tag="25.11.01", asset_name="DonkeySimMac.zip", asset_size=123456789)
        self.calls = []

    def get_installed(self):
        self.calls.append(("get_installed",))
        return self.installed

    def fetch_latest(self, force=False):
        self.calls.append(("fetch_latest",))
        return self.latest_info

    def check_for_update(self, force=False):
        self.calls.append(("check_for_update",))
        return SimpleNamespace(update_available=True, latest=self.latest_info)

    def stage_update(self, release=None):
        self.calls.append(("stage_update",))

    def apply_staged_update(self, pid=None, timeout=None):
        self.calls.append(("apply_staged_update", pid))
        self.installed = self.latest_info.tag
        return SimpleNamespace(tag=self.latest_info.tag)

    def install_update(self, release=None, pid=None, timeout=None):
        self.calls.append(("install_update", pid))

    def use_version(self, tag):
        self.calls.append(("use_version", tag))
        return SimpleNamespace(tag=tag)

    def clean(self, keep_versions=2):
        self.calls.append(("clean", keep_versions))
        return ["25.10.06"]


class NoUpdateFakeUpdater(FakeUpdater):
    def check_for_update(self, force=False):
        self.calls.append(("check_for_update",))
        return False


class UpdateCheckFakeUpdater(FakeUpdater):
    def check_for_update(self, force=False):
        self.calls.append(("check_for_update",))
        return SimpleNamespace(update_available=True, latest=SimpleNamespace(tag="25.11.01"))


class FailingFetchUpdater(FakeUpdater):
    def fetch_latest(self, force=False):
        self.calls.append(("fetch_latest",))
        raise RuntimeError("connection refused")


class FailingCheckUpdater(FakeUpdater):
    def check_for_update(self, force=False):
        self.calls.append(("check_for_update",))
        raise RuntimeError("connection refused")


class FailingStageUpdater(FakeUpdater):
    def stage_update(self, release=None):
        self.calls.append(("stage_update",))
        raise OSError("download failed")


class RejectingUseUpdater(FakeUpdater):
    def use_version(self, tag):
        self.calls.append(("use_version", tag))
        raise ValueError("unknown version")


@pytest.fixture
def fake_updater(monkeypatch):
    FakeUpdater.instances = []
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", FakeUpdater)
    yield FakeUpdater


@pytest.fixture
def managed_exe(tmp_path):
    exe = tmp_path / "DonkeySimMac"
    exe.write_text("#!/bin/sh\n")
    return str(exe)


@pytest.fixture
def spawn_recorder(monkeypatch):
    calls = []

    def fake_spawn(exe, host, port, headless):
        calls.append((exe, host, port, headless))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(simulator_update, "_spawn_managed", fake_spawn)
    return calls


def test_status_reports_installed_and_latest(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: "/tmp/sim-root/current/DonkeySimMac")
    rc = simulator_update.main(["status"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert ("get_installed",) in updater.calls
    assert ("fetch_latest",) in updater.calls
    assert ("stage_update",) not in updater.calls
    assert ("apply_staged_update", None) not in updater.calls
    out = capsys.readouterr().out
    assert "25.10.06" in out
    assert "25.11.01" in out
    assert "/tmp/sim-root/current/DonkeySimMac" in out


def test_status_network_failure_exits_3(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", FailingFetchUpdater)
    rc = simulator_update.main(["status"])
    assert rc == 3
    out = capsys.readouterr().out
    assert "installed:  25.10.06" in out
    assert "latest:     unavailable" in out


def test_check_reports_update_available(fake_updater, capsys):
    rc = simulator_update.main(["check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Update available: 25.10.06 -> 25.11.01" in out
    assert ("check_for_update",) in FakeUpdater.instances[-1].calls


def test_check_reports_up_to_date(monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", NoUpdateFakeUpdater)
    rc = simulator_update.main(["check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Up to date." in out


def test_check_reports_versions_from_update_check(monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", UpdateCheckFakeUpdater)
    rc = simulator_update.main(["check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Update available: 25.10.06 -> 25.11.01" in out


def test_check_network_failure_exits_3(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", FailingCheckUpdater)
    rc = simulator_update.main(["check"])
    assert rc == 3
    assert "connection refused" in capsys.readouterr().err


def test_dry_run_prints_summary_without_downloading(fake_updater, tmp_path, capsys):
    root = str(tmp_path / "sim-root")
    rc = simulator_update.main(["update", "--dry-run", "--root", root])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert ("fetch_latest",) in updater.calls
    for call in updater.calls:
        assert call[0] not in ("stage_update", "apply_staged_update", "install_update")
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split(":", 1)[0] for line in lines] == ["repo", "current", "latest", "asset", "size", "target"]
    assert "DonkeySimMac.zip" in lines[3]
    assert "123,456,789 bytes (117.7 MiB)" in lines[4]
    assert lines[5].endswith(os.path.join("25.11.01"))


def test_dry_run_network_failure_exits_3(fake_updater, monkeypatch):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", FailingFetchUpdater)
    rc = simulator_update.main(["update", "--dry-run"])
    assert rc == 3


@pytest.mark.parametrize(
    "extra", [["--version", "25.10.06"], ["--pid", "1"], ["--apply-after-exit", "--pid", "1"], ["--restart"]]
)
def test_dry_run_conflicts_are_rejected(fake_updater, extra):
    rc = simulator_update.main(["update", "--dry-run"] + extra)
    assert rc == 2
    assert FakeUpdater.instances == []


def test_apply_after_exit_requires_pid(fake_updater, capsys):
    rc = simulator_update.main(["update", "--apply-after-exit"])
    assert rc == 2
    assert FakeUpdater.instances == []
    assert "--pid" in capsys.readouterr().err


def test_update_default_stages_and_applies(fake_updater, capsys):
    rc = simulator_update.main(["update"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert updater.calls.index(("stage_update",)) < updater.calls.index(("apply_staged_update", None))
    out = capsys.readouterr().out
    assert "Installed version 25.11.01" in out


def test_update_skips_download_when_already_up_to_date(monkeypatch, capsys):
    FakeUpdater.instances = []
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", NoUpdateFakeUpdater)
    rc = simulator_update.main(["update"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert ("stage_update",) not in updater.calls
    assert ("apply_staged_update", None) not in updater.calls
    assert "Already up to date." in capsys.readouterr().out


def test_apply_after_exit_waits_before_relaunch_when_up_to_date(monkeypatch, managed_exe, spawn_recorder, capsys):
    FakeUpdater.instances = []
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", NoUpdateFakeUpdater)
    waits = []
    monkeypatch.setattr(
        simulator_update,
        "wait_for_pid",
        lambda pid, timeout=None: waits.append(pid) or True,
    )
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: managed_exe)

    rc = simulator_update.main(["update", "--apply-after-exit", "--pid", "123", "--restart"])

    assert rc == 0
    assert waits == [123, 123]
    assert spawn_recorder == [(managed_exe, "0.0.0.0", 9091, False)]
    assert "Already up to date." in capsys.readouterr().out


def test_update_version_switches_to_installed_version(fake_updater):
    rc = simulator_update.main(["update", "--version", "25.10.06"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert ("use_version", "25.10.06") in updater.calls
    assert ("stage_update",) not in updater.calls


def test_update_with_pid_stages_but_does_not_apply(fake_updater, capsys):
    rc = simulator_update.main(["update", "--pid", "123"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert ("stage_update",) in updater.calls
    assert ("apply_staged_update", None) not in updater.calls
    out = capsys.readouterr().out
    assert "staged but not applied" in out
    assert "PID 123" in out


def test_update_network_failure_exits_3(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", FailingStageUpdater)
    rc = simulator_update.main(["update"])
    assert rc == 3
    updater = FailingStageUpdater.instances[-1]
    assert ("apply_staged_update", None) not in updater.calls
    assert "download failed" in capsys.readouterr().err


def test_apply_after_exit_waits_for_pid_then_applies(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "wait_for_pid", lambda pid, timeout=None: False)
    rc = simulator_update.main(["update", "--apply-after-exit", "--pid", "123"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert updater.calls.index(("stage_update",)) < updater.calls.index(("apply_staged_update", 123))
    out = capsys.readouterr().out
    assert "Waiting for process 123 to exit" in out
    assert "Installed version 25.11.01" in out


def test_apply_after_exit_notes_already_dead_pid(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "wait_for_pid", lambda pid, timeout=None: True)
    rc = simulator_update.main(["update", "--apply-after-exit", "--pid", "123"])
    assert rc == 0
    assert ("apply_staged_update", 123) in FakeUpdater.instances[-1].calls
    assert "Process 123 is not running" in capsys.readouterr().out


def test_use_passes_version_through(fake_updater):
    rc = simulator_update.main(["use", "25.10.06"])
    assert rc == 0
    assert ("use_version", "25.10.06") in FakeUpdater.instances[-1].calls


def test_use_unknown_version_exits_2(fake_updater, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "SimulatorUpdater", RejectingUseUpdater)
    rc = simulator_update.main(["use", "nope"])
    assert rc == 2
    assert "unknown version" in capsys.readouterr().err


def test_clean_reports_removed_versions(fake_updater, capsys):
    rc = simulator_update.main(["clean"])
    assert rc == 0
    assert ("clean", 2) in FakeUpdater.instances[-1].calls
    out = capsys.readouterr().out
    assert "Removed 1 old version(s): 25.10.06." in out


def test_spawn_managed_uses_unity_headless_flags(monkeypatch):
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(pid=2468)

    monkeypatch.setattr(simulator_update.subprocess, "Popen", fake_popen)

    process = simulator_update._spawn_managed("/sim/donkey_sim", "127.0.0.1", 9999, True)

    assert process.pid == 2468
    cmd, kwargs = calls[0]
    assert cmd == ["/sim/donkey_sim", "--host", "127.0.0.1", "--port", "9999", "-batchmode", "-nographics"]
    if os.name == "nt":
        assert kwargs["creationflags"]
    else:
        assert kwargs["start_new_session"] is True


def test_launch_spawns_managed_executable_with_defaults(managed_exe, spawn_recorder, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: managed_exe)
    rc = simulator_update.main(["launch"])
    assert rc == 0
    assert spawn_recorder == [(managed_exe, "0.0.0.0", 9091, False)]
    assert "pid 12345" in capsys.readouterr().out


def test_launch_honors_host_port_headless(managed_exe, spawn_recorder, monkeypatch):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: managed_exe)
    rc = simulator_update.main(["launch", "--host", "127.0.0.1", "--port", "9999", "--headless"])
    assert rc == 0
    assert spawn_recorder == [(managed_exe, "127.0.0.1", 9999, True)]


def test_launch_without_installation_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: None)
    rc = simulator_update.main(["launch"])
    assert rc == 2
    assert "not installed" in capsys.readouterr().err


def test_restart_starts_new_version_with_launch_args(fake_updater, managed_exe, spawn_recorder, monkeypatch, capsys):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: managed_exe)
    rc = simulator_update.main(["update", "--restart", "--host", "127.0.0.1", "--port", "9999", "--headless"])
    assert rc == 0
    assert spawn_recorder == [(managed_exe, "127.0.0.1", 9999, True)]
    assert ("apply_staged_update", None) in FakeUpdater.instances[-1].calls
    assert "pid 12345" in capsys.readouterr().out


def test_restart_after_exit_also_relaunches(fake_updater, managed_exe, spawn_recorder, monkeypatch):
    monkeypatch.setattr(simulator_update, "resolve_managed_executable", lambda root: managed_exe)
    monkeypatch.setattr(simulator_update, "wait_for_pid", lambda pid, timeout=None: True)
    rc = simulator_update.main(["update", "--apply-after-exit", "--pid", "123", "--restart"])
    assert rc == 0
    assert spawn_recorder == [(managed_exe, "0.0.0.0", 9091, False)]


def test_restart_with_pid_requires_apply_after_exit(fake_updater):
    rc = simulator_update.main(["update", "--pid", "123", "--restart"])
    assert rc == 2
    assert FakeUpdater.instances == []


def test_version_conflicts_with_apply_after_exit(fake_updater):
    rc = simulator_update.main(["update", "--version", "25.10.06", "--apply-after-exit", "--pid", "123"])
    assert rc == 2
    assert FakeUpdater.instances == []


def test_global_options_are_forwarded_to_updater(fake_updater, tmp_path):
    root = str(tmp_path / "sim-root")
    rc = simulator_update.main(["status", "--repo", "someone/donkey-fork", "--root", root, "--timeout", "12.5"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert updater.repo == "someone/donkey-fork"
    assert updater.root == root
    assert updater.timeout == 12.5
    assert updater.platform in ("darwin", "linux", "win32")
    assert callable(updater.progress)


def test_global_options_before_subcommand_are_forwarded_to_updater(fake_updater, tmp_path):
    root = str(tmp_path / "sim-root-before")
    rc = simulator_update.main(["--repo", "someone/donkey-fork", "--root", root, "--timeout", "9.5", "status"])
    assert rc == 0
    updater = FakeUpdater.instances[-1]
    assert updater.repo == "someone/donkey-fork"
    assert updater.root == root
    assert updater.timeout == 9.5


def test_progress_reports_percentage(fake_updater, capsys):
    rc = simulator_update.main(["check"])
    assert rc == 0
    progress = FakeUpdater.instances[-1].progress
    progress(21, 50)
    progress(50, 50)
    err = capsys.readouterr().err
    assert "\rDownloading...  42%" in err
    assert "\rDownloading... 100%\n" in err


def test_format_size():
    assert simulator_update._format_size(123456789) == "117.7 MiB"
    assert simulator_update._format_size(512) == "512 B"


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as exc_info:
        simulator_update.main(["frobnicate"])
    assert exc_info.value.code == 2


def test_missing_command_exits_2():
    with pytest.raises(SystemExit) as exc_info:
        simulator_update.main([])
    assert exc_info.value.code == 2
