#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for managed simulator updates in DonkeyEnv / DonkeyUnityProcess.

Everything is mocked: SimulatorUpdater / resolve_managed_executable and
subprocess.Popen are never called for real, so these tests never touch the
network or the managed simulator directory on disk.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import types
from unittest import mock

import pytest

# core.simulator_updater.py is implemented in parallel with this integration.
# Until it lands, register a stub module so donkey_env's lazy import (and the
# patches below, which target the source module) can resolve.
try:
    from gym_donkeycar.core.simulator_updater import SimulatorUpdater, resolve_managed_executable  # noqa: F401
except ImportError:
    _updater_stub = types.ModuleType("gym_donkeycar.core.simulator_updater")
    _updater_stub.SimulatorUpdater = None
    _updater_stub.resolve_managed_executable = None
    sys.modules["gym_donkeycar.core.simulator_updater"] = _updater_stub

from gym_donkeycar.envs.donkey_env import DonkeyEnv, supply_defaults
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess

UPDATER_MODULE = "gym_donkeycar.core.simulator_updater"
SIM_PATH = "/managed-sim/DonkeySimMac.app/Contents/MacOS/donkey_sim"
SIM_ROOT = "/tmp/fake-managed-simulators"


@pytest.fixture
def env_mocks():
    with mock.patch("gym_donkeycar.envs.donkey_env.DonkeyUnitySimContoller") as viewer_cls, mock.patch(
        "gym_donkeycar.envs.donkey_env.DonkeyUnityProcess"
    ) as proc_cls:
        viewer_cls.return_value.get_sensor_size.return_value = (8, 8, 3)
        viewer_cls.return_value.get_obs_size.return_value = 4
        yield viewer_cls, proc_cls


def make_conf(**overrides):
    conf = {
        "exe_path": "managed",
        "host": "127.0.0.1",
        "port": 9091,
        "start_delay": 0.0,
        "sim_update_mode": "off",
        "sim_update_root": SIM_ROOT,
    }
    conf.update(overrides)
    return conf


# ---------------------------------------------------------------------------
# supply_defaults
# ---------------------------------------------------------------------------


def test_supply_defaults_adds_sim_update_keys():
    with mock.patch.dict(os.environ, {}, clear=True):
        conf = {}
        supply_defaults(conf)
    assert conf["sim_update_mode"] == "off"
    assert conf["sim_update_repo"] == "tawnkramer/gym-donkeycar"
    assert conf["sim_update_root"] == os.path.expanduser("~/.donkeycar/simulators")
    assert conf["sim_update_cache_hours"] == 24
    assert conf["sim_update_keep_versions"] == 2


# ---------------------------------------------------------------------------
# DonkeyEnv managed simulator handling
# ---------------------------------------------------------------------------


def test_managed_off_applies_staged_then_resolves_and_starts(env_mocks):
    _, proc_cls = env_mocks
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ) as resolve:
        updater_cls.return_value.staged_path.exists.return_value = True
        order = mock.Mock()
        order.attach_mock(updater_cls.return_value.apply_staged_update, "apply_staged")
        order.attach_mock(resolve, "resolve")

        env = DonkeyEnv(level="warehouse", conf=make_conf())

        # staged update is applied before the executable is resolved
        assert order.mock_calls == [mock.call.apply_staged(), mock.call.resolve(root=SIM_ROOT)]
        updater_cls.assert_called_once_with(repo="tawnkramer/gym-donkeycar", root=SIM_ROOT, cache_seconds=86400.0)
        updater_cls.return_value.clean.assert_called_once_with(keep_versions=2)
        assert env.updater is updater_cls.return_value
        assert proc_cls.return_value.start.call_args.args[0] == SIM_PATH
        updater_cls.return_value.stage_update.assert_not_called()
        updater_cls.return_value.check_for_update.assert_not_called()
        # mode off: no background worker
        assert env.update_thread is None
        env.close()


@pytest.mark.parametrize("mode", ["off", "check", "auto"])
def test_managed_not_installed_raises_and_never_downloads(env_mocks, mode):
    _, proc_cls = env_mocks
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=None
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        with pytest.raises(RuntimeError, match="donkey-sim-update update"):
            DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode=mode))
    # no implicit download, no simulator process
    proc_cls.return_value.start.assert_not_called()
    updater_cls.return_value.stage_update.assert_not_called()
    updater_cls.return_value.check_for_update.assert_not_called()


def test_managed_invalid_mode_raises(env_mocks):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        with pytest.raises(ValueError, match="sim_update_mode"):
            DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="nope"))
    updater_cls.assert_not_called()


def test_check_mode_warns_about_new_version_in_background(env_mocks, caplog):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        updater_cls.return_value.check_for_update.return_value = types.SimpleNamespace(
            update_available=True, latest=types.SimpleNamespace(tag="v25.10.07")
        )
        with caplog.at_level(logging.WARNING, logger="gym_donkeycar.envs.donkey_env"):
            env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="check"))
            assert env.update_thread is not None
            assert env.update_thread.daemon is True
            env.update_thread.join(timeout=5)
        assert not env.update_thread.is_alive()
        updater_cls.return_value.check_for_update.assert_called_once_with(force=False)
        updater_cls.return_value.stage_update.assert_not_called()
        assert any("v25.10.07" in rec.message for rec in caplog.records)
        # check mode never stages, so close has nothing to apply
        env.close()
        assert updater_cls.return_value.apply_staged_update.call_count == 0


def test_check_mode_silent_when_up_to_date(env_mocks, caplog):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        updater_cls.return_value.check_for_update.return_value = types.SimpleNamespace(
            update_available=False, latest=types.SimpleNamespace(tag="v25.10.06")
        )
        with caplog.at_level(logging.WARNING, logger="gym_donkeycar.envs.donkey_env"):
            env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="check"))
            env.update_thread.join(timeout=5)
        assert not any(rec.message for rec in caplog.records)
        env.close()


def test_cache_hours_zero_forces_fresh_lookup(env_mocks):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        updater_cls.return_value.check_for_update.return_value = types.SimpleNamespace(update_available=False, latest=None)
        env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="check", sim_update_cache_hours=0))
        env.update_thread.join(timeout=5)
        updater_cls.return_value.check_for_update.assert_called_once_with(force=True)
        assert updater_cls.call_args.kwargs["cache_seconds"] == 0.0
        env.close()


def test_auto_with_cache_hours_zero_checks_fresh_and_stages_release(env_mocks):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        release = types.SimpleNamespace()
        updater_cls.return_value.check_for_update.return_value = types.SimpleNamespace(update_available=True, latest=release)
        updater_cls.return_value.stage_update.return_value = {"tag": "v25.10.07"}
        env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="auto", sim_update_cache_hours=0))
        env.update_thread.join(timeout=5)
        updater_cls.return_value.check_for_update.assert_called_once_with(force=True)
        updater_cls.return_value.stage_update.assert_called_once_with(release=release)
        env.close()


def test_auto_does_not_download_when_already_up_to_date(env_mocks):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        updater_cls.return_value.check_for_update.return_value = types.SimpleNamespace(update_available=False, latest=None)
        env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="auto"))
        env.update_thread.join(timeout=5)

        assert env.update_staged is False
        updater_cls.return_value.stage_update.assert_not_called()

        env.close()


def test_auto_stage_completed_then_close_applies_staged(env_mocks):
    viewer_cls, proc_cls = env_mocks
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        updater_cls.return_value.stage_update.return_value = {"tag": "v25.10.07"}
        latest = updater_cls.return_value.check_for_update.return_value.latest
        env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="auto"))
        assert env.update_thread is not None
        assert env.update_thread.daemon is True
        env.update_thread.join(timeout=5)
        assert not env.update_thread.is_alive()
        assert env.update_staged is True
        updater_cls.return_value.stage_update.assert_called_once_with(release=latest)

        events = []
        viewer_cls.return_value.quit.side_effect = lambda: events.append("viewer.quit")
        proc_cls.return_value.quit.side_effect = lambda: events.append("proc.quit")
        updater_cls.return_value.apply_staged_update.side_effect = lambda: events.append("apply_staged")
        updater_cls.return_value.clean.side_effect = lambda keep_versions: events.append("clean")

        env.close()
        # viewer first, then proc, then the completed staged update
        assert events == ["viewer.quit", "proc.quit", "apply_staged", "clean"]
        assert updater_cls.return_value.apply_staged_update.call_count == 1
        updater_cls.return_value.clean.assert_called_once_with(keep_versions=2)

        # close is idempotent: nothing else runs on the second call
        env.close()
        assert events == ["viewer.quit", "proc.quit", "apply_staged", "clean"]
        assert updater_cls.return_value.apply_staged_update.call_count == 1


def test_auto_stage_still_running_at_close_leaves_it_for_next_start(env_mocks):
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable", return_value=SIM_PATH
    ):
        updater_cls.return_value.staged_path.exists.return_value = False
        gate = threading.Event()

        def slow_stage(release=None):
            gate.wait(timeout=10)
            return {"tag": "v25.10.07"}

        updater_cls.return_value.stage_update.side_effect = slow_stage
        env = DonkeyEnv(level="warehouse", conf=make_conf(sim_update_mode="auto"))
        # gym.make returned while staging is still in flight: never blocks on it
        assert env.update_thread.is_alive()
        assert env.update_staged is False

        t0 = time.monotonic()
        env.close()
        elapsed = time.monotonic() - t0
        # close gave up after a short grace period instead of waiting
        assert elapsed < 2.0
        assert updater_cls.return_value.apply_staged_update.call_count == 0
        assert env.update_thread.is_alive()

        gate.set()
        env.update_thread.join(timeout=5)
        # the worker still records its result; the next startup applies it
        assert env.update_staged is True


def test_normal_exe_path_never_touches_updater(env_mocks):
    _, proc_cls = env_mocks
    with mock.patch(f"{UPDATER_MODULE}.SimulatorUpdater") as updater_cls, mock.patch(
        f"{UPDATER_MODULE}.resolve_managed_executable"
    ) as resolve:
        env = DonkeyEnv(
            level="warehouse",
            conf=make_conf(exe_path="/plain/path/donkey_sim", sim_update_mode="auto"),
        )
    updater_cls.assert_not_called()
    resolve.assert_not_called()
    assert env.updater is None
    assert env.update_thread is None
    assert proc_cls.return_value.start.call_args.args[0] == "/plain/path/donkey_sim"
    env.close()


# ---------------------------------------------------------------------------
# DonkeyUnityProcess lifecycle
# ---------------------------------------------------------------------------


class FakeProc:
    """Minimal Popen stand-in used to exercise the quit escalation path."""

    def __init__(self, pid: int = 4242, block_on_terminate: bool = False):
        self.pid = pid
        self.calls = []
        self.returncode = None
        self.block_on_terminate = block_on_terminate
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        self.terminated = True
        if not self.block_on_terminate:
            self.returncode = 0

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.block_on_terminate and self.terminated and not self.killed:
            raise subprocess.TimeoutExpired(cmd=["donkey_sim"], timeout=timeout)
        if self.killed:
            self.returncode = -9
        return self.returncode

    def kill(self):
        self.calls.append("kill")
        self.killed = True


def test_proc_start_records_pid_and_running():
    fake = FakeProc()
    with mock.patch("gym_donkeycar.envs.donkey_proc.subprocess.Popen", return_value=fake), mock.patch(
        "gym_donkeycar.envs.donkey_proc.os.path.exists", return_value=True
    ):
        proc = DonkeyUnityProcess()
        proc.start("/fake/donkey_sim", port=9091)
    assert proc.pid == 4242
    assert proc.is_running is True


def test_proc_remote_and_missing_path_leave_no_process():
    proc = DonkeyUnityProcess()
    proc.start("remote")
    assert proc.proc1 is None
    assert proc.pid is None
    assert proc.is_running is False
    proc.quit()  # no-op without a process

    with mock.patch("gym_donkeycar.envs.donkey_proc.os.path.exists", return_value=False):
        proc.start("/missing/donkey_sim")
    assert proc.proc1 is None
    assert proc.is_running is False


def test_proc_quit_terminates_and_waits_gracefully():
    fake = FakeProc()
    with mock.patch("gym_donkeycar.envs.donkey_proc.subprocess.Popen", return_value=fake), mock.patch(
        "gym_donkeycar.envs.donkey_proc.os.path.exists", return_value=True
    ):
        proc = DonkeyUnityProcess()
        proc.start("/fake/donkey_sim")
        proc.quit()
    assert fake.calls == ["terminate", ("wait", 5.0)]
    assert fake.killed is False
    assert proc.proc1 is None
    assert proc.pid is None
    assert proc.is_running is False
    proc.quit()  # idempotent: a second quit is a no-op


def test_proc_quit_kills_only_after_grace_timeout():
    fake = FakeProc(block_on_terminate=True)
    with mock.patch("gym_donkeycar.envs.donkey_proc.subprocess.Popen", return_value=fake), mock.patch(
        "gym_donkeycar.envs.donkey_proc.os.path.exists", return_value=True
    ):
        proc = DonkeyUnityProcess()
        proc.start("/fake/donkey_sim")
        proc.quit(timeout=0.25)
    assert fake.calls == ["terminate", ("wait", 0.25), "kill", ("wait", None)]
    assert proc.proc1 is None
    assert proc.pid is None
    assert proc.is_running is False
