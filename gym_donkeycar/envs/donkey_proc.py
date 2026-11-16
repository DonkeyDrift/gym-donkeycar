"""
file: donkey_proc.py
author: Felix Yu
date: 2018-09-12
"""

import os
import subprocess
from typing import Optional, Sequence


class DonkeyUnityProcess:
    def __init__(self):
        self.proc1 = None
        self.pid = None

    # ------ Launch Unity Env ----------- #

    def start(
        self,
        sim_path: str,
        host: str = "0.0.0.0",
        port: int = 9091,
        headless: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        keep_graphics: bool = False,
    ):
        if sim_path == "remote":
            return

        if not os.path.exists(sim_path):
            print(sim_path, "does not exist. you must start sim manually.")
            return

        # 按端口区分日志文件名,多 sim 实例同目录并行时不再互相覆盖
        port_args = ["--port", str(port), "--host", str(host), "-logFile", f"unitylog_{port}.txt"]

        if headless:
            # keep_graphics: -batchmode 但保留图形设备——离屏渲染,无窗口/无遮挡休眠,
            # 相机可用;纯状态量训练用 -nographics 更省
            port_args += ["-batchmode"]
            if not keep_graphics:
                port_args += ["-nographics"]

        if extra_args:
            port_args += list(extra_args)

        # Launch Unity environment
        self.proc1 = subprocess.Popen([sim_path] + port_args)
        self.pid = self.proc1.pid

        print("donkey subprocess started")

    @property
    def is_running(self) -> bool:
        return self.proc1 is not None and self.proc1.poll() is None

    def quit(self, timeout: float = 5.0) -> None:
        """
        Shutdown unity environment.

        Terminate gracefully first and wait up to `timeout` seconds; only
        escalate to kill if the process does not exit in time. Calling quit
        without a started process (or twice) is a no-op.
        """
        proc = self.proc1
        if proc is None:
            return

        print("closing donkey sim subprocess")

        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            self.proc1 = None
            self.pid = None
