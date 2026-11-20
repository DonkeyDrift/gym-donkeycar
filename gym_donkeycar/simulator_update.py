#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command line interface for the managed Donkey simulator.

The ``donkey-sim-update`` console script downloads, verifies, installs and
launches the Donkey Unity simulator from its upstream GitHub releases.

Exit codes:
    0  success
    2  invalid arguments or unexpected installation state
    3  network, download or verification failure
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

from gym_donkeycar.core.simulator_updater import (
    DEFAULT_SIMULATOR_UPDATE_REPO,
    SimulatorUpdater,
    resolve_managed_executable,
    wait_for_pid,
)

DEFAULT_REPO = DEFAULT_SIMULATOR_UPDATE_REPO
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".donkeycar", "simulators")
DEFAULT_TIMEOUT = 15.0

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9091

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _detect_platform() -> str:
    """Return the ``sys.platform`` value selecting the platform release asset."""
    if sys.platform == "cygwin":
        return "win32"
    return sys.platform


class _Progress:
    """Report download progress as a percentage on a single stderr line."""

    def __init__(self, stream, label: str = "Downloading") -> None:
        self._stream = stream
        self._label = label
        self._last_percent = -1

    def __call__(self, received: int, total: int = 0) -> None:
        if total <= 0:
            if received > 0:
                self._stream.write("\r{}... {} bytes".format(self._label, received))
                self._stream.flush()
            return
        percent = min(100, received * 100 // total)
        if percent != self._last_percent:
            self._last_percent = percent
            self._stream.write("\r{}... {:3d}%".format(self._label, percent))
            self._stream.flush()
        if percent >= 100:
            self._last_percent = -1
            self._stream.write("\n")
            self._stream.flush()


def _paint(text: str, code: str, enabled: bool) -> str:
    return code + text + _RESET if enabled else text


def _colors_enabled(args, stream) -> bool:
    return not args.no_color and stream.isatty()


def _error(args, message: str, code: int = 2) -> int:
    print(_paint("error: ", _RED, _colors_enabled(args, sys.stderr)) + message, file=sys.stderr)
    return code


def _ok(args, message: str) -> None:
    print(_paint(message, _GREEN, _colors_enabled(args, sys.stdout)))


def _field(info, *names):
    """Read a field from a dict-like or object-like updater result value."""
    if info is None:
        return None
    if isinstance(info, str):
        return info
    for name in names:
        if isinstance(info, dict):
            if name in info and info[name] is not None:
                return info[name]
        else:
            value = getattr(info, name, None)
            if value is not None:
                return value
    return None


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return "{} B".format(int(value))
            return "{:.1f} {}".format(value, unit)
        value /= 1024.0
    return str(num_bytes)


def _installed_version(updater):
    installed = updater.get_installed()
    if isinstance(installed, str):
        return installed or None
    return _field(installed, "tag", "version", "tag_name", "installed")


def _make_updater(args):
    progress = _Progress(sys.stderr)
    return SimulatorUpdater(
        repo=args.repo,
        root=args.root,
        platform=_detect_platform(),
        timeout=args.timeout,
        progress=progress,
        prerelease=getattr(args, "prerelease", False),
    )


def _announce_wait(pid: int) -> None:
    if wait_for_pid(pid, timeout=0):
        print("Process {} is not running.".format(pid))
    else:
        print("Waiting for process {} to exit...".format(pid))


def _spawn_managed(exe, host: str, port: int, headless: bool):
    cmd = [str(exe), "--host", host, "--port", str(port)]
    if headless:
        cmd.extend(["-batchmode", "-nographics"])
    if os.name == "nt":
        return subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return subprocess.Popen(cmd, start_new_session=True)


def _restart(args) -> int:
    try:
        exe = resolve_managed_executable(args.root)
    except Exception as exc:
        return _error(args, "cannot resolve the managed executable: {}.".format(exc))
    if not exe:
        return _error(args, "managed simulator is not installed; run 'donkey-sim-update update' first.")
    try:
        proc = _spawn_managed(exe, args.host, args.port, args.headless)
    except Exception as exc:
        return _error(args, "failed to start the managed simulator: {}.".format(exc))
    _ok(args, "Started managed simulator (pid {}).".format(proc.pid))
    return 0


def _cmd_status(args) -> int:
    updater = _make_updater(args)
    try:
        installed = updater.get_installed()
    except Exception as exc:
        return _error(args, "failed to read the installed version: {}.".format(exc))
    latest = None
    fetch_error = None
    try:
        latest = _field(updater.fetch_latest(), "tag", "tag_name", "version")
    except Exception as exc:
        fetch_error = exc
    try:
        exe = resolve_managed_executable(args.root)
    except Exception:
        exe = None
    print("channel:    {}".format("prerelease" if getattr(args, "prerelease", False) else "stable"))
    print("repo:       {}".format(args.repo))
    print("root:       {}".format(args.root))
    print("installed:  {}".format(_field(installed, "tag", "version") or "none"))
    print("executable: {}".format(exe or "not installed"))
    if fetch_error is not None:
        print("latest:     unavailable ({})".format(fetch_error))
        return 3
    print("latest:     {}".format(latest or "unknown"))
    return 0


def _cmd_check(args) -> int:
    updater = _make_updater(args)
    try:
        installed = _installed_version(updater)
    except Exception as exc:
        if getattr(args, "json", False):
            channel = "prerelease" if getattr(args, "prerelease", False) else "stable"
            print(
                json.dumps(
                    {"check_ok": False, "update_available": False, "channel": channel, "error": str(exc)},
                    separators=(",", ":"),
                )
            )
            return 3
        return _error(args, "failed to read the installed version: {}.".format(exc))
    try:
        result = updater.check_for_update()
    except Exception as exc:
        if getattr(args, "json", False):
            channel = "prerelease" if getattr(args, "prerelease", False) else "stable"
            print(
                json.dumps(
                    {"check_ok": False, "update_available": False, "channel": channel, "error": str(exc)},
                    separators=(",", ":"),
                )
            )
            return 3
        return _error(args, "failed to check for updates: {}.".format(exc), 3)
    available = False
    latest = None
    if isinstance(result, bool):
        available = result
    elif isinstance(result, (tuple, list)):
        latest = result[-1] if result else None
        available = bool(latest and str(latest) != str(installed))
    else:
        available = bool(getattr(result, "update_available", False))
        latest = _field(getattr(result, "latest", None), "tag", "version", "tag_name")
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "check_ok": True,
                    "update_available": bool(available),
                    "installed": installed,
                    "latest": latest,
                    "channel": "prerelease" if getattr(args, "prerelease", False) else "stable",
                    "source": "github",
                },
                separators=(",", ":"),
            )
        )
        return 0
    print("installed: {}".format(installed or "none"))
    if latest:
        print("latest:           {}".format(latest))
    if available:
        if latest and str(latest) != str(installed):
            print("Update available: {} -> {}".format(installed or "none", latest))
        else:
            print("Update available.")
    else:
        print("Up to date.")
    return 0


def _run_dry_run(args) -> int:
    updater = _make_updater(args)
    try:
        current = _installed_version(updater)
    except Exception:
        current = None
    try:
        info = updater.fetch_latest()
    except Exception as exc:
        return _error(args, "failed to fetch the latest release: {}.".format(exc), 3)
    latest = _field(info, "tag", "tag_name", "version")
    asset = _field(info, "asset_name", "asset", "filename", "name")
    size = _field(info, "asset_size", "size", "size_bytes", "bytes")
    target = _field(info, "target", "target_dir", "target_path")
    print("repo:    {}".format(args.repo))
    print("current: {}".format(current or "none"))
    print("latest:  {}".format(latest or "unknown"))
    print("asset:   {}".format(asset or "unknown"))
    if isinstance(size, (int, float)):
        size_text = "{:,} bytes ({})".format(int(size), _format_size(int(size)))
    else:
        size_text = str(size) if size is not None else "unknown"
    print("size:    {}".format(size_text))
    print("target:  {}".format(target or os.path.join(args.root, str(latest or "unknown"))))
    return 0


def _update_available(updater):
    result = updater.check_for_update()
    available = bool(getattr(result, "update_available", result))
    latest = getattr(result, "latest", None)
    return available, latest


def _cmd_update(args) -> int:
    if args.dry_run:
        for option in ("version", "apply_after_exit", "pid", "restart"):
            if getattr(args, option):
                return _error(args, "--dry-run cannot be combined with --version, --pid, --apply-after-exit or --restart.")
        return _run_dry_run(args)
    if args.apply_after_exit and not args.pid:
        return _error(args, "--apply-after-exit requires --pid.")
    if args.apply_after_exit and args.version:
        return _error(args, "--version cannot be combined with --apply-after-exit.")
    if args.pid and args.restart and not args.apply_after_exit:
        return _error(args, "--restart with --pid requires --apply-after-exit.")
    updater = _make_updater(args)
    if args.version:
        try:
            print("Switching to version {}...".format(args.version))
            record = updater.use_version(args.version)
        except Exception as exc:
            return _error(args, "cannot switch to version {}: {}.".format(args.version, exc))
        _ok(args, "Switched to version {}.".format(_field(record, "tag", "version") or args.version))
    elif args.apply_after_exit:
        try:
            available, release = _update_available(updater)
            if not available:
                print("Already up to date.")
                if not args.restart:
                    return 0
                _announce_wait(args.pid)
                if not wait_for_pid(args.pid):
                    return _error(args, "process {} did not exit; not restarting.".format(args.pid))
                return _restart(args)
            print("Staging update...")
            updater.stage_update(release=release)
        except Exception as exc:
            return _error(args, "update failed: {}.".format(exc), 3)
        _announce_wait(args.pid)
        try:
            print("Applying update...")
            record = updater.apply_staged_update(pid=args.pid)
        except Exception as exc:
            return _error(args, "update failed: {}.".format(exc))
        _ok(args, "Installed version {}.".format(_field(record, "tag", "version") or "latest"))
    elif args.pid:
        try:
            available, release = _update_available(updater)
            if not available:
                print("Already up to date.")
                return _restart(args) if args.restart else 0
            print("Staging update...")
            updater.stage_update(release=release)
        except Exception as exc:
            return _error(args, "update failed: {}.".format(exc), 3)
        _ok(args, "Update staged but not applied (simulator may be running, PID {}).".format(args.pid))
        print("Rerun 'donkey-sim-update update' after stopping it, or use --apply-after-exit.")
        return 0
    else:
        try:
            available, release = _update_available(updater)
            if not available:
                print("Already up to date.")
                return _restart(args) if args.restart else 0
            print("Updating to the latest version...")
            updater.stage_update(release=release)
        except Exception as exc:
            return _error(args, "update failed: {}.".format(exc), 3)
        try:
            print("Applying update...")
            record = updater.apply_staged_update()
        except Exception as exc:
            return _error(args, "update failed: {}.".format(exc))
        _ok(args, "Installed version {}.".format(_field(record, "tag", "version") or "latest"))
    if args.restart:
        return _restart(args)
    return 0


def _cmd_use(args) -> int:
    updater = _make_updater(args)
    try:
        record = updater.use_version(args.version)
    except Exception as exc:
        return _error(args, "cannot use version {}: {}.".format(args.version, exc))
    _ok(args, "Switched to version {}.".format(_field(record, "tag", "version") or args.version))
    return 0


def _cmd_clean(args) -> int:
    updater = _make_updater(args)
    try:
        removed = updater.clean()
    except Exception as exc:
        return _error(args, "clean failed: {}.".format(exc))
    if isinstance(removed, (list, tuple)):
        if removed:
            _ok(args, "Removed {} old version(s): {}.".format(len(removed), ", ".join(removed)))
        else:
            _ok(args, "Nothing to clean.")
    else:
        _ok(args, "Cleaned the managed installation.")
    return 0


def _cmd_launch(args) -> int:
    return _restart(args)


def _add_common_arguments(parser: argparse.ArgumentParser, include_defaults: bool = True) -> None:
    repo_default = DEFAULT_REPO if include_defaults else argparse.SUPPRESS
    root_default = DEFAULT_ROOT if include_defaults else argparse.SUPPRESS
    timeout_default = DEFAULT_TIMEOUT if include_defaults else argparse.SUPPRESS
    no_color_default = False if include_defaults else argparse.SUPPRESS
    parser.add_argument(
        "--repo",
        default=repo_default,
        help="upstream GitHub repository, owner/name" + (" (default: %s)" % DEFAULT_REPO if include_defaults else ""),
    )
    parser.add_argument(
        "--root",
        default=root_default,
        help="managed install directory" + (" (default: %s)" % DEFAULT_ROOT if include_defaults else ""),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=timeout_default,
        metavar="SECONDS",
        help="network timeout in seconds" + (" (default: %s)" % DEFAULT_TIMEOUT if include_defaults else ""),
    )
    parser.add_argument("--no-color", action="store_true", default=no_color_default, help="disable colored output")
    prerelease_default = False if include_defaults else argparse.SUPPRESS
    parser.add_argument(
        "--prerelease",
        action="store_true",
        default=prerelease_default,
        help="resolve updates from the newest pre-release instead of the latest stable release",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="donkey-sim-update",
        description="Download, update and launch the managed Donkey simulator.",
    )
    _add_common_arguments(parser)

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    status = sub.add_parser("status", help="show installed/latest versions and the managed executable")
    _add_common_arguments(status, include_defaults=False)
    status.set_defaults(handler=_cmd_status)

    check = sub.add_parser("check", help="check whether a newer version is available")
    _add_common_arguments(check, include_defaults=False)
    check.add_argument("--json", action="store_true", help="write a machine-readable JSON result to stdout")
    check.set_defaults(handler=_cmd_check)

    update = sub.add_parser("update", aliases=["install"], help="download, verify, stage and apply an update")
    _add_common_arguments(update, include_defaults=False)
    update.add_argument("--version", metavar="VERSION", help="switch to this already installed version instead of the latest")
    update.add_argument("--dry-run", action="store_true", help="only print repo/current/latest/asset/size/target")
    update.add_argument(
        "--apply-after-exit", action="store_true", help="wait for the process given by --pid to exit, then apply"
    )
    update.add_argument("--pid", type=int, metavar="PID", help="PID of a running simulator; do not apply while it runs")
    update.add_argument("--restart", action="store_true", help="start the newly installed version after applying")
    update.add_argument("--host", default=DEFAULT_HOST, help="simulator host for --restart (default: %(default)s)")
    update.add_argument("--port", type=int, default=DEFAULT_PORT, help="simulator port for --restart (default: %(default)s)")
    update.add_argument("--headless", action="store_true", help="start the simulator without graphics for --restart")
    update.set_defaults(handler=_cmd_update)

    use = sub.add_parser("use", help="switch to an already installed version")
    _add_common_arguments(use, include_defaults=False)
    use.add_argument("version", metavar="VERSION", help="version to switch to")
    use.set_defaults(handler=_cmd_use)

    clean = sub.add_parser("clean", help="remove staging files and old versions")
    _add_common_arguments(clean, include_defaults=False)
    clean.set_defaults(handler=_cmd_clean)

    launch = sub.add_parser("launch", help="start the managed simulator")
    _add_common_arguments(launch, include_defaults=False)
    launch.add_argument("--host", default=DEFAULT_HOST, help="simulator host (default: %(default)s)")
    launch.add_argument("--port", type=int, default=DEFAULT_PORT, help="simulator port (default: %(default)s)")
    launch.add_argument("--headless", action="store_true", help="start without graphics")
    launch.set_defaults(handler=_cmd_launch)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ``donkey-sim-update`` command line interface."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
