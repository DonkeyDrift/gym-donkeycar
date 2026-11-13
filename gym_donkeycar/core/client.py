"""
SDClient

A base class for interacting with the sdsim simulator as server.
The server will create on vehicle per client connection. The client
will then interact by createing json message to send to the server.
The server will reply with telemetry and other status messages in an
asynchronous manner.

Author: Tawn Kramer
"""

import json
import logging
import select
import socket
import time
from threading import Thread, current_thread
from typing import Any, Dict

from .util import replace_float_notation

logger = logging.getLogger(__name__)

# 分段耗时累计(进程内诊断):recv/parse/telemetry 三段,驱动进程观测用。
# key: recv_wait( select+sleep 等待)、recv_xfer(recv+字符串拼接)、
#      parse(json+分发)、telemetry(图像解码+字段处理,主要在 on_telemetry)
TIMING = {"recv_wait": 0.0, "recv_xfer": 0.0, "parse": 0.0, "telemetry": 0.0, "n": 0}


def _t(key, dt):
    TIMING[key] += dt


class SDClient:
    def __init__(self, host: str, port: int, poll_socket_sleep_time: float = 0.001):
        self.msg = None
        self.host = host
        self.port = port
        self.poll_socket_sleep_sec = poll_socket_sleep_time
        self.th = None

        # the aborted flag will be set when we have detected a problem with the socket
        # that we can't recover from.
        self.aborted = False
        self.s = None
        self.connect()

    def connect(self) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(5.0)
        # 大接收缓冲 + 禁 Nagle，避免高频遥测因消费抖动反压模拟器发送线程。
        try:
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        except OSError:
            pass
        try:
            self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        # connecting to the server
        logger.info("connecting to %s:%d " % (self.host, self.port))
        try:
            self.s.connect((self.host, self.port))
        except ConnectionRefusedError:
            self.s.close()
            raise (
                Exception(
                    "Could not connect to server. Is it running? "
                    "If you specified 'remote', then you must start it manually."
                )
            )
        except OSError:
            self.s.close()
            raise

        # time.sleep(pause_on_create)
        self.do_process_msgs = True
        self.th = Thread(target=self.proc_msg, args=(self.s,), daemon=True)
        self.th.start()

    def send(self, m: str) -> None:
        self.msg = m

    def send_now(self, msg: str) -> None:
        logger.debug("send_now:" + msg)
        self.s.sendall(msg.encode("utf-8"))

    def on_msg_recv(self, j: Dict[str, Any]) -> None:
        logger.debug("got:" + j["msg_type"])

    def is_connected(self) -> bool:
        """TCP 会话仍健康，且接收线程正在运行。"""
        return (
            not self.aborted
            and self.do_process_msgs
            and self.s is not None
            and self.s.fileno() != -1
            and self.th is not None
            and self.th.is_alive()
        )

    def stop(self) -> None:
        # 幂等且可重入：on_abort 可能就在读线程里调用 stop()。
        self.do_process_msgs = False
        self.aborted = True
        if self.s is not None:
            try:
                self.s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.s.close()
        if self.th is not None and self.th is not current_thread():
            self.th.join(timeout=2.0)

    def proc_msg(self, sock: socket.socket) -> None:  # noqa: C901
        """
        This is the thread message loop to process messages.
        We will send any message that is queued via the self.msg variable
        when our socket is in a writable state.
        And we will read any messages when it's in a readable state and then
        call self.on_msg_recv with the json object message.
        """
        try:
            sock.setblocking(False)
        except OSError:
            self.stop()
            return
        inputs = [sock]
        outputs = [sock]
        localbuffer = ""

        while self.do_process_msgs:
            # without this sleep, I was getting very consistent socket errors
            # on Windows. Perhaps we don't need this sleep on other platforms.
            time.sleep(self.poll_socket_sleep_sec)
            t0 = time.perf_counter()
            try:
                # test our socket for readable, writable states.
                readable, writable, exceptional = select.select(inputs, outputs, inputs, 0.1)
                t1 = time.perf_counter()
                _t("recv_wait", t1 - t0)

                for s in readable:
                    try:
                        data = s.recv(1024 * 256)
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
                        logger.warning("socket recv failed: %s", e)
                        self.aborted = True
                        self.do_process_msgs = False
                        try:
                            self.on_msg_recv({"msg_type": "aborted"})
                        except Exception:
                            logger.exception("on_abort 通知失败")
                        return

                    if not data:
                        # FIN 也必须收敛，否则 select/recv 会持续空转。
                        logger.warning("socket closed by peer (EOF)")
                        self.aborted = True
                        self.do_process_msgs = False
                        try:
                            self.on_msg_recv({"msg_type": "aborted"})
                        except Exception:
                            logger.exception("on_abort 通知失败")
                        return

                    # we don't technically need to convert from bytes to string
                    # for json.loads, but we do need a string in order to do
                    # the split by \n newline char. This seperates each json msg.
                    data = data.decode("utf-8")

                    localbuffer += data
                    t2 = time.perf_counter()
                    _t("recv_xfer", t2 - t1)

                    n0 = localbuffer.find("{")
                    n1 = localbuffer.rfind("}\n")
                    if n1 >= 0 and 0 <= n0 < n1:  # there is at least one message :
                        msgs = localbuffer[n0 : n1 + 1].split("\n")
                        localbuffer = localbuffer[n1:]

                        for m in msgs:
                            if not self.do_process_msgs:
                                break
                            if len(m) <= 2:
                                continue
                            # Replace comma with dots for floats
                            # useful when using unity in a language different from English
                            m = replace_float_notation(m)
                            try:
                                j = json.loads(m)
                            except Exception as e:
                                logger.error("Exception:" + str(e))
                                logger.error("json: " + m)
                                continue

                            if "msg_type" not in j:
                                logger.error("Warning expected msg_type field")
                                logger.error("json: " + m)
                                continue
                            else:
                                t3 = time.perf_counter()
                                self.on_msg_recv(j)
                                _t("telemetry", time.perf_counter() - t3)
                                TIMING["n"] += 1
                        _t("parse", time.perf_counter() - t2)

                for s in writable:
                    if self.msg is not None:
                        logger.debug("sending " + self.msg)
                        s.sendall(self.msg.encode("utf-8"))
                        self.msg = None

                if len(exceptional) > 0:
                    raise ConnectionError("simulator socket exception")

            except Exception as e:
                if self.do_process_msgs:
                    logger.warning("simulator connection lost: %s", e)
                    self.aborted = True
                    self.do_process_msgs = False
                    try:
                        self.on_msg_recv({"msg_type": "aborted"})
                    except Exception:
                        logger.exception("simulator disconnect callback failed")
                self.stop()
                break
