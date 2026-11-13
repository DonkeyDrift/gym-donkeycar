#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `DonkeyUnitySimContoller.wait_until_loaded` resilience.

Background defect: wait_until_loaded() only polled handler.loaded, so a
connection that died between TCP accept and the car_loaded message left the
client aborted but loaded forever False -- gym.make (and any caller such as
the dgym reconnect worker) blocked permanently inside the wait loop.
"""

import base64
import io
import json
import logging
import socket
import struct
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from gym_donkeycar.core.client import SDClient
from gym_donkeycar.envs.donkey_sim import (
    DonkeyUnitySimContoller,
    DonkeyUnitySimHandler,
)


def _telemetry_frame(w: int = 32, h: int = 24) -> bytes:
    """One minimal sim-protocol telemetry line with a tiny PNG image."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    payload = {
        "msg_type": "telemetry",
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
    }
    return (json.dumps(payload) + "\n").encode()


class _SimServer:
    """Minimal fake sim. Modes:
    - car_loaded: send car_loaded, hold the connection open, stay silent
    - telemetry: send car_loaded, then stream telemetry forever
    - silent_rst: send nothing, reset the socket after kill_after seconds
    """

    def __init__(self, mode: str, kill_after: float = 0.2, telemetry_period: float = 0.03):
        assert mode in ("car_loaded", "telemetry", "silent_rst")
        self.mode = mode
        self.kill_after = kill_after
        self.telemetry_period = telemetry_period
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self._conn = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        self._conn = conn
        if self.mode in ("car_loaded", "telemetry"):
            try:
                conn.sendall(b'{"msg_type": "car_loaded"}\n')
            except OSError:
                return
            if self.mode == "telemetry":
                try:
                    conn.settimeout(self.telemetry_period)
                    while True:
                        try:
                            data = conn.recv(4096)
                            if not data:
                                return
                        except socket.timeout:
                            pass
                        conn.sendall(_telemetry_frame())
                except OSError:
                    return
            # car_loaded mode: hold the connection open until torn down
            try:
                conn.settimeout(30.0)
                while True:
                    data = conn.recv(4096)
                    if not data:
                        return
            except OSError:
                return
        else:
            time.sleep(self.kill_after)
            # SO_LINGER(1, 0) + close => RST, never sending car_loaded
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self.sock.close()


def _conf(port: int):
    return {
        "host": "127.0.0.1",
        "port": port,
        "level": "generated_track",
        "max_cte": 8.0,
        "cam_resolution": (24, 32, 3),
        "log_level": logging.CRITICAL,
    }


def _handler_with_mock_client(timeout=None):
    """Handler wired to a minimal mock client without aborted/do_process_msgs
    attributes (SimpleNamespace), like lightweight test doubles."""
    conf = {
        "level": "generated_track",
        "max_cte": 8.0,
        "cam_resolution": (24, 32, 3),
    }
    if timeout is not None:
        conf["observe_timeout"] = timeout
    handler = DonkeyUnitySimHandler(conf)
    handler.client = SimpleNamespace(
        is_connected=lambda: True,
        send_now=lambda message: None,
        queue_message=lambda message: None,
    )
    return handler


def _bounded_observe(handler, join_timeout=0.5):
    """Run observe() in a daemon thread; never hangs the test process."""
    errors = []

    def observe():
        try:
            handler.observe()
        except Exception as e:  # noqa: BLE001 - propagate to main thread
            errors.append(e)

    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    worker.join(timeout=join_timeout)
    stopped = not worker.is_alive()
    if not stopped:
        # release a legacy unbounded wait so teardown stays clean
        handler.time_received += 1
        worker.join(timeout=join_timeout)
    return stopped, errors


def test_wait_until_loaded_raises_when_connection_dies_before_car_loaded():
    """Dead connection before car_loaded must raise ConnectionError, not wait
    forever (old implementation spun here permanently)."""
    srv = _SimServer("silent_rst")
    controller = None
    errors = []
    try:
        controller = DonkeyUnitySimContoller(conf=_conf(srv.port))

        def call_wait():
            try:
                controller.wait_until_loaded()
            except Exception as e:  # noqa: BLE001 - propagate to main thread
                errors.append(e)

        th = threading.Thread(target=call_wait, daemon=True)
        th.start()
        # daemon thread + bounded join: the old implementation stays stuck
        # here (RED) without hanging the test process itself
        th.join(timeout=5.0)
        assert not th.is_alive(), (
            "wait_until_loaded still blocks after the connection died " "before car_loaded (permanent wait, old behavior)"
        )
        assert len(errors) == 1, f"expected exactly one exception: {errors}"
        assert isinstance(errors[0], ConnectionError), errors
        assert controller.client.aborted is True
    finally:
        if controller is not None:
            controller.quit()
        srv.close()


def test_wait_until_loaded_returns_when_sim_reports_loaded():
    """Healthy path: car_loaded arrives, wait_until_loaded returns promptly."""
    srv = _SimServer("car_loaded")
    controller = None
    errors = []
    try:
        controller = DonkeyUnitySimContoller(conf=_conf(srv.port))

        def call_wait():
            try:
                controller.wait_until_loaded()
            except Exception as e:  # noqa: BLE001 - propagate to main thread
                errors.append(e)

        th = threading.Thread(target=call_wait, daemon=True)
        th.start()
        th.join(timeout=5.0)
        assert not th.is_alive(), "wait_until_loaded hung on a healthy sim"
        assert errors == [], f"healthy path should not raise: {errors}"
        assert controller.handler.loaded is True
    finally:
        if controller is not None:
            controller.quit()
        srv.close()


def test_observe_times_out_on_silent_stream():
    """observe() must raise ConnectionError within conf["observe_timeout"] when
    the connection is alive but sends no telemetry (silent sim)."""
    srv = _SimServer("car_loaded")
    controller = None
    errors = []
    try:
        conf = _conf(srv.port)
        conf["observe_timeout"] = 0.5
        controller = DonkeyUnitySimContoller(conf=conf)

        def call_observe():
            try:
                controller.handler.observe()
            except Exception as e:  # noqa: BLE001 - propagate to main thread
                errors.append(e)

        th = threading.Thread(target=call_observe, daemon=True)
        t0 = time.monotonic()
        th.start()
        # daemon thread + bounded join: the un-timeouted wait stays stuck here
        # (RED) without hanging the test process itself
        th.join(timeout=5.0)
        elapsed = time.monotonic() - t0
        assert not th.is_alive(), "observe() hung on a connected-but-silent sim " "(no observe_timeout support)"
        assert len(errors) == 1, f"expected exactly one exception: {errors}"
        assert isinstance(errors[0], TimeoutError), errors
        assert elapsed >= 0.5, f"raised before the timeout: {elapsed:.2f}s"
        assert elapsed < 3.0, f"observe_timeout not honored: {elapsed:.2f}s"
        # the timeout is a signal to the caller, not a hard abort: the client
        # itself must stay alive so the caller decides how to tear down
        assert controller.client.aborted is False
        assert controller.client.do_process_msgs is True
    finally:
        if controller is not None:
            controller.quit()
        srv.close()


def test_observe_returns_frames_with_timeout_configured():
    """observe_timeout must not break the healthy path: streaming frames keep
    observe() returning normally, well inside the configured bound."""
    srv = _SimServer("telemetry")
    controller = None
    errors = []
    try:
        conf = _conf(srv.port)
        conf["observe_timeout"] = 2.0
        controller = DonkeyUnitySimContoller(conf=conf)

        def call_observe():
            try:
                for _ in range(3):
                    controller.handler.observe()
            except Exception as e:  # noqa: BLE001 - propagate to main thread
                errors.append(e)

        th = threading.Thread(target=call_observe, daemon=True)
        th.start()
        th.join(timeout=5.0)
        assert not th.is_alive(), "observe() hung on a healthy streaming sim"
        assert errors == [], f"healthy path should not raise: {errors}"
    finally:
        if controller is not None:
            controller.quit()
        srv.close()


def test_connect_timeout_closes_socket():
    """A connect timeout must close the socket and re-raise TimeoutError."""
    with patch("gym_donkeycar.core.client.socket.socket") as factory:
        sock = factory.return_value
        sock.connect.side_effect = TimeoutError("connect timed out")
        with pytest.raises(TimeoutError):
            SDClient("127.0.0.1", 1)
        sock.close.assert_called_once()


def test_connect_refused_closes_socket():
    """Connect failures must not leak the socket either."""
    with patch("gym_donkeycar.core.client.socket.socket") as factory:
        sock = factory.return_value
        sock.connect.side_effect = ConnectionRefusedError()
        with pytest.raises(Exception, match="Could not connect to server"):
            SDClient("127.0.0.1", 1)
        sock.close.assert_called_once()


def test_observe_exits_after_disconnect():
    handler = _handler_with_mock_client(timeout=0.1)
    handler.on_disconnect()
    stopped, errors = _bounded_observe(handler)
    assert stopped, "observe remained stuck after disconnect"
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)


def test_observe_exits_after_explicit_close():
    handler = _handler_with_mock_client(timeout=0.1)
    handler.on_close()
    stopped, errors = _bounded_observe(handler)
    assert stopped, "observe remained stuck after close"
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)


def test_observe_tolerates_client_without_state_attributes():
    """A client lacking aborted/do_process_msgs must not cause AttributeError;
    observe falls back to the configured timeout instead."""
    handler = _handler_with_mock_client(timeout=0.05)
    stopped, errors = _bounded_observe(handler)
    assert stopped, "silent connection must not wait forever"
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError), errors


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_observe_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="observe_timeout"):
        _handler_with_mock_client(timeout)


def test_last_telemetry_ts_tracks_real_telemetry_and_reset_keeps_it():
    handler = _handler_with_mock_client()
    assert handler.last_telemetry_ts == 0
    handler.on_telemetry(json.loads(_telemetry_frame().decode()))
    received = handler.last_telemetry_ts
    assert received > 0
    with patch("gym_donkeycar.envs.donkey_sim.time.sleep"):
        handler.reset()
    assert handler.last_telemetry_ts == received


def test_wait_until_loaded_timeout_stops_client_first():
    """Silent scene load within observe_timeout: stop the client, then raise
    TimeoutError (and never hang gym.make forever)."""
    controller = DonkeyUnitySimContoller.__new__(DonkeyUnitySimContoller)
    controller.handler = _handler_with_mock_client(timeout=0.01)
    stopped = []
    controller.client = SimpleNamespace(is_connected=lambda: True, stop=lambda: stopped.append(True))
    with pytest.raises(TimeoutError):
        controller.wait_until_loaded()
    assert stopped == [True]
