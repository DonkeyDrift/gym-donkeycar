"""
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
"""

import base64
import logging
import math
import os
import time
import types
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # 无 opencv 的环境回退 PIL 解码
    cv2 = None

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient


def _decode_jpeg_to_rgb(jpg_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> RGB ndarray。

    cv2 路径优先:解码期释放 GIL 且比 PIL 快——车端进程里采集线程要与 WebRTC
    编码、车辆主循环抢 GIL,PIL 慢路径会把遥测消费速率拖低,TCP 管道反压回
    模拟器发送线程,源帧率被钉在 ~52Hz(见 SDClient.connect 的大缓冲注释)。
    """
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.open(BytesIO(jpg_bytes))
    return np.asarray(image)

logger = logging.getLogger(__name__)


# Math helpers added by CireNeikual (222464)
def euler_to_quat(e):
    cx = np.cos(e[0] * 0.5)
    sx = np.sin(e[0] * 0.5)
    cy = np.cos(e[1] * 0.5)
    sy = np.sin(e[1] * 0.5)
    cz = np.cos(e[2] * 0.5)
    sz = np.sin(e[2] * 0.5)

    x = sz * cx * cy - cz * sx * sy
    y = cz * sx * cy + sz * cx * sy
    z = cz * cx * sy - sz * sx * cy
    w = cz * cx * cy + sz * sx * sy

    return [x, y, z, w]


def cross(v0, v1):
    return [v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]]


def rotate_vec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [v[0] + uv[0] + uuv[0], v[1] + uv[1] + uuv[1], v[2] + uv[2] + uuv[2]]


class DonkeyUnitySimContoller:
    def __init__(self, conf: Dict[str, Any]):
        logger.setLevel(conf["log_level"])

        self.conf = conf
        self.address = (conf["host"], conf["port"])

        self.handler = DonkeyUnitySimHandler(conf=conf)

        self.client = SimClient(self.address, self.handler)

    def set_car_config(
        self,
        body_style: str,
        body_rgb: Tuple[int, int, int],
        car_name: str,
        font_size: int,
    ) -> None:
        self.handler.send_car_config(body_style, body_rgb, car_name, font_size)

    def set_cam_config(self, **kwargs) -> None:
        self.handler.send_cam_config(**kwargs)

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.handler.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.handler.set_episode_over_fn(ep_over_fn)

    def wait_until_loaded(self) -> None:
        time.sleep(0.1)
        deadline = time.time() + 15.0
        while not self.handler.loaded:
            client = getattr(self.handler, "client", None)
            if client is None or not getattr(client, "do_process_msgs", True):
                # 连接已停止：不再无限等待场景加载（避免调用线程被困死）
                break
            if time.time() > deadline:
                # 模拟器接了连接但一直不出场景：超时放行，交给上层
                # 看门狗/健康检查决定是否重建，不让调用线程永久卡死
                logger.warning("wait_until_loaded timeout, continue anyway")
                break
            logger.warning("waiting for sim to start..")
            time.sleep(1.0)
        logger.info("sim started!")

        # RL 同步步进:车加载完成后进同步模式(每条 control 推进一步并回一帧)
        if self.conf.get("sync_mode", False):
            msg = {
                "msg_type": "step_mode",
                "step_mode": "synchronous",
                "time_step": str(self.conf.get("sync_time_step", 0.05)),
            }
            self.handler.blocking_send(msg)
            logger.info("synchronous step mode enabled, time_step=%s", msg["time_step"])

    def reset(self, seed=None) -> None:
        self.handler.reset(seed=seed)

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.handler.get_sensor_size()

    def get_obs_size(self):
        return self.handler.get_obs_size()

    def take_action(self, action: np.ndarray):
        self.handler.take_action(action)

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.handler.observe()

    def quit(self) -> None:
        # 同步模式 best-effort 退出:恢复模拟器全局物理后再断开,发不出去也无所谓
        if self.conf.get("sync_mode", False):
            try:
                self.handler.blocking_send({"msg_type": "step_mode", "step_mode": "asynchronous"})
            except Exception:
                logger.debug("failed to send async step_mode on quit", exc_info=True)
        self.client.stop()

    def exit_scene(self) -> None:
        self.handler.send_exit_scene()

    def render(self, mode: str) -> None:
        pass

    def is_game_over(self) -> bool:
        return self.handler.is_game_over()

    def calc_reward(self, done: bool) -> float:
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    OBS_VECTOR_SIZE = 8

    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.SceneToLoad = conf["level"]
        self.loaded = False
        self.max_cte = conf["max_cte"]
        self.timer = FPSTimer()

        # RL 协议扩展配置(sdsandbox P0-P2,均有缺省,旧 conf 不受影响)
        self.sync_mode = conf.get("sync_mode", False)
        self.sync_time_step = conf.get("sync_time_step", 0.05)
        self.state_obs = conf.get("state_obs", False)
        self.spawn_cfg = conf.get("spawn_cfg", None)
        self.done_on_flip = conf.get("done_on_flip", True)
        # 训练专用:连接中断或等不到新遥测时抛异常(毒化可见),而非静默返回陈旧观测
        self.raise_on_stale = conf.get("raise_on_stale", False)
        self.done_on_stuck = conf.get("done_on_stuck", True)
        self.stuck_time_thresh = conf.get("stuck_time_thresh", 2.5)

        # sensor size - height, width, depth
        self.camera_img_size = conf["cam_resolution"]
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.missed_checkpoint = False
        self.dq = False
        self.over = False
        self.client = None
        self.fns = {
            "telemetry": self.on_telemetry,
            "scene_selection_ready": self.on_scene_selection_ready,
            "scene_names": self.on_recv_scene_names,
            "car_loaded": self.on_car_loaded,
            "cross_start": self.on_cross_start,
            "race_start": self.on_race_start,
            "race_stop": self.on_race_stop,
            "DQ": self.on_DQ,
            "ping": self.on_ping,
            "aborted": self.on_abort,
            "missed_checkpoint": self.on_missed_checkpoint,
            "need_car_config": self.on_need_car_config,
            "collision_with_starting_line": self.on_collision_with_starting_line,
        }
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []

        # car in Unity lefthand coordinate system: roll is Z, pitch is X and yaw is Y
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # variables required for lidar points decoding into array format
        self.lidar_deg_per_sweep_inc = 1
        self.lidar_num_sweep_levels = 1
        self.lidar_deg_ang_delta = 1

        self.last_lap_time = 0.0
        self.current_lap_time = 0.0
        self.starting_line_index = -1
        self.lap_count = 0

        # sdsandbox 新遥测字段(RL):缺省保证 determine_episode_over 可用
        # steps_since_reset 必须在此初始化:车加载后 sim 立刻推异步遥测,
        # 首个 on_telemetry 早于任何 reset()——只在 _clear_episode_state 初始化
        # 会让接收线程在首帧抛 AttributeError 死亡,observe() 退化为
        # 瞬时陈旧观测(假步进,2026-10-18 事故)。
        self.steps_since_reset = 0
        self.collision_impulse = 0.0
        self.flipped = False
        self.stuck_time = 0.0
        self.lap = 0
        self.frame_id = 0
        self.active_node = 0
        self.total_nodes = 1

    def on_connect(self, client: SimClient) -> None:  # pytype: disable=signature-mismatch
        logger.debug("socket connected")
        self.client = client

    def on_disconnect(self) -> None:
        logger.debug("socket disconnected")
        self.client = None

    def on_abort(self, message: Dict[str, Any]) -> None:
        if self.client is not None:
            self.client.stop()
        # 清掉 handler 的 client 引用：上层（dgym）健康探测与 observe()
        # 的自旋逃逸都依赖它感知失联
        self.on_disconnect()

    def on_need_car_config(self, message: Dict[str, Any]) -> None:
        logger.info("on need car config")
        self.loaded = True
        self.send_config(self.conf)

    def on_collision_with_starting_line(self, message: Dict[str, Any]) -> None:
        if self.current_lap_time == 0.0:
            self.current_lap_time = message["timeStamp"]
            self.starting_line_index = message["starting_line_index"]
        elif self.starting_line_index == message["starting_line_index"]:
            time_at_crossing = message["timeStamp"]
            self.last_lap_time = float(time_at_crossing - self.current_lap_time)
            self.current_lap_time = time_at_crossing
            self.lap_count += 1
            lap_msg = f"New lap time: {round(self.last_lap_time, 2)} seconds"
            logger.info(lap_msg)

    @staticmethod
    def extract_keys(dict_: Dict[str, Any], list_: List[str]) -> Dict[str, Any]:
        return_dict = {}
        for key in list_:
            if key in dict_:
                return_dict[key] = dict_[key]
        return return_dict

    def send_config(self, conf: Dict[str, Any]) -> None:
        if "degPerSweepInc" in conf:
            raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

        logger.info("sending car config.")
        # both ways work, car_config shouldn't interfere with other config, so keeping the two alternative
        self.set_car_config(conf)
        if "car_config" in conf.keys():
            self.set_car_config(conf["car_config"])
            logger.info("done sending car config.")

        if "cam_config" in conf.keys():
            cam_config = self.extract_keys(
                conf["cam_config"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                    "fps",
                ],
            )
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")

        if "cam_config_b" in conf.keys():
            cam_config_b = self.extract_keys(
                conf["cam_config_b"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                    "fps",
                ],
            )
            self.send_cam_config(**cam_config_b, msg_type="cam_config_b")
            logger.info(f"done sending cam config B. {cam_config_b}")
            self.image_array_b = np.zeros(self.camera_img_size)

        if "lidar_config" in conf.keys():
            if "degPerSweepInc" in conf:
                raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

            lidar_config = self.extract_keys(
                conf["lidar_config"],
                [
                    "deg_per_sweep_inc",
                    "deg_ang_down",
                    "deg_ang_delta",
                    "num_sweeps_levels",
                    "max_range",
                    "noise",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                ],
            )
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")

        # what follows is needed in order not to break older conf

        cam_config = self.extract_keys(
            conf,
            [
                "img_w",
                "img_h",
                "img_d",
                "img_enc",
                "fov",
                "fish_eye_x",
                "fish_eye_y",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "fps",
            ],
        )
        if cam_config != {}:
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")
            logger.warning("""This way of passing cam_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'cam_config'.
                Example: GYM_CONF = {'cam_config':""" + str(cam_config) + "}")

        lidar_config = self.extract_keys(
            conf,
            [
                "deg_per_sweep_inc",
                "deg_ang_down",
                "deg_ang_delta",
                "num_sweeps_levels",
                "max_range",
                "noise",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
            ],
        )
        if lidar_config != {}:
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")
            logger.warning("""This way of passing lidar_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'lidar_config'.
                Example: GYM_CONF = {'lidar_config':""" + str(lidar_config) + "}")

    def set_car_config(self, conf: Dict[str, Any]) -> None:
        if "body_style" in conf:
            self.send_car_config(
                conf["body_style"],
                conf["body_rgb"],
                conf["car_name"],
                conf["font_size"],
            )

    def set_racer_bio(self, conf: Dict[str, Any]) -> None:
        if "bio" in conf:
            self.send_racer_bio(
                conf["racer_name"],
                conf["car_name"],
                conf["bio"],
                conf["country"],
                conf["guid"],
            )

    def on_recv_message(self, message: Dict[str, Any]) -> None:
        if "msg_type" not in message:
            logger.warn("expected msg_type field")
            return
        msg_type = message["msg_type"]
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logger.warning(f"unknown message type {msg_type}")

    # ------- Env interface ---------- #

    def reset(self, seed: Optional[int] = None) -> None:
        logger.debug("reseting")
        self.timer.reset()

        if self.sync_mode:
            # 同步模式:先清状态,再发 reset_car,自旋等模拟器的 reset 应答帧
            # (timeScale=0 下没有异步流,必须先等到应答再让上层 observe)
            self._clear_episode_state()
            self.send_reset_car(seed=seed)
            deadline = time.time() + 10.0
            last = self.time_received
            while self.time_received == last:
                client = self.client
                if client is None or not getattr(client, "do_process_msgs", True):
                    break
                if time.time() > deadline:
                    logger.warning("reset: timeout waiting for sync reset answer frame")
                    break
                time.sleep(0.001)
        else:
            # 异步模式保持原行为:先发复位,睡 1s 让模拟器落地,再清本地状态
            self.send_reset_car(seed=seed)
            time.sleep(1)
            self._clear_episode_state()

    def _clear_episode_state(self) -> None:
        self.steps_since_reset = 0
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.over = False
        self.missed_checkpoint = False
        self.dq = False
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.lap_count = 0

        # car
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # 新遥测字段一并复位
        self.collision_impulse = 0.0
        self.flipped = False
        self.stuck_time = 0.0
        self.lap = 0
        self.frame_id = 0
        self.active_node = 0
        self.total_nodes = 1

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.camera_img_size

    def get_obs_size(self):
        if self.state_obs:
            return (self.OBS_VECTOR_SIZE,)
        return self.camera_img_size

    def take_action(self, action: np.ndarray) -> None:
        self.send_control(action[0], action[1])

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        spin_deadline = time.time() + 15.0
        while self.last_received == self.time_received:
            client = self.client
            if client is None or not getattr(client, "do_process_msgs", True):
                # 连接已停止：不再等待新遥测，立即返回旧观测。
                # 否则调用方（车端 step 线程）会在此永久自旋——连接被
                # 重建后新 env 无人驱动，表现为模拟器永久掉线。
                if self.raise_on_stale:
                    raise ConnectionError("sim connection lost during observe()")
                break
            if time.time() > spin_deadline:
                # 连接活着但迟迟没有新遥测：也返回旧观测，让调用方的
                # 看门狗/健康检查接管（重建连接），不让线程永久卡死
                if self.raise_on_stale:
                    raise TimeoutError("no fresh telemetry within 15s during observe()")
                break
            time.sleep(0.001)

        self.last_received = self.time_received

        if self.state_obs:
            # 状态量观测(无图像遥测模式):8 维定长向量,分量归一化到约 [-1,1]
            observation = np.array(
                [
                    self.cte / 2.0,
                    self.speed / 10.0,
                    self.forward_vel / 10.0,
                    self.yaw / 180.0,
                    self.gyro_z / 90.0,
                    self.active_node / max(self.total_nodes, 1),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            self.last_obs = observation
        else:
            observation = self.image_array

        done = self.is_game_over()
        reward = self.calc_reward(done)

        info = {
            "pos": (self.x, self.y, self.z),
            "cte": self.cte,
            "speed": self.speed,
            "forward_vel": self.forward_vel,
            "hit": self.hit,
            "gyro": (self.gyro_x, self.gyro_y, self.gyro_z),
            "accel": (self.accel_x, self.accel_y, self.accel_z),
            "vel": (self.vel_x, self.vel_y, self.vel_z),
            "lidar": (self.lidar),
            "car": (self.roll, self.pitch, self.yaw),
            "last_lap_time": self.last_lap_time,
            "lap_count": self.lap_count,
            "collision_impulse": self.collision_impulse,
            "flipped": self.flipped,
            "stuck_time": self.stuck_time,
            "lap": self.lap,
            "frame_id": self.frame_id,
            "state_obs": self.state_obs,
        }

        # Add the second image to the dict
        if self.image_array_b is not None:
            info["image_b"] = self.image_array_b

        # self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self) -> bool:
        return self.over

    # ------ RL interface ----------- #

    def set_reward_fn(self, reward_fn: Callable[[], float]):
        """
        allow users to set their own reward function
        """
        self.calc_reward = types.MethodType(reward_fn, self)
        logger.debug("custom reward fn set.")

    def calc_reward(self, done: bool) -> float:
        # Normalization factor, real max speed is around 30
        # but only attained on a long straight line
        # max_speed = 10

        if done:
            return -1.0

        if self.cte > self.max_cte:
            return -1.0

        # Collision
        if self.hit != "none":
            return -2.0

        # going fast close to the center of lane yeilds best reward
        if self.forward_vel > 0.0:
            return (1.0 - (math.fabs(self.cte) / self.max_cte)) * self.forward_vel

        # in reverse, reward doesn't have centering term as this can result in some exploits
        return self.forward_vel

    # ------ Socket interface ----------- #

    def on_telemetry(self, message: Dict[str, Any]) -> None:
        # img_enc="none" 时模拟器省略 image 键,仅解码存在的图像
        if "image" in message:
            img_string = message["image"]
            self.image_array = _decode_jpeg_to_rgb(base64.b64decode(img_string))

        # always update the image_array as the observation loop will hang if not changing.
        self.time_received = time.time()
        self.steps_since_reset += 1

        if "image_b" in message:
            img_string_b = message["image_b"]
            self.image_array_b = _decode_jpeg_to_rgb(base64.b64decode(img_string_b))

        if "pos_x" in message:
            self.x = message["pos_x"]
            self.y = message["pos_y"]
            self.z = message["pos_z"]

        if "speed" in message:
            self.speed = message["speed"]

        e = [self.pitch * np.pi / 180.0, self.yaw * np.pi / 180.0, self.roll * np.pi / 180.0]
        q = euler_to_quat(e)

        forward = rotate_vec(q, [0.0, 0.0, 1.0])

        # dot
        self.forward_vel = forward[0] * self.vel_x + forward[1] * self.vel_y + forward[2] * self.vel_z

        if "gyro_x" in message:
            self.gyro_x = message["gyro_x"]
            self.gyro_y = message["gyro_y"]
            self.gyro_z = message["gyro_z"]
        if "accel_x" in message:
            self.accel_x = message["accel_x"]
            self.accel_y = message["accel_y"]
            self.accel_z = message["accel_z"]
        if "vel_x" in message:
            self.vel_x = message["vel_x"]
            self.vel_y = message["vel_y"]
            self.vel_z = message["vel_z"]

        if "roll" in message:
            self.roll = message["roll"]
            self.pitch = message["pitch"]
            self.yaw = message["yaw"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in message:
            self.cte = message["cte"]

        # sdsandbox 新遥测字段(RL):始终存在,仍按存在性守卫兼容旧模拟器
        if "collision_impulse" in message:
            self.collision_impulse = message["collision_impulse"]
        if "flipped" in message:
            self.flipped = message["flipped"]
        if "stuck_time" in message:
            self.stuck_time = message["stuck_time"]
        if "lap" in message:
            self.lap = message["lap"]
        if "frame_id" in message:
            self.frame_id = message["frame_id"]
        if "activeNode" in message:
            self.active_node = message["activeNode"]
        if "totalNodes" in message:
            self.total_nodes = message["totalNodes"]

        if "lidar" in message:
            self.lidar = self.process_lidar_packet(message["lidar"])

        # don't update hit once session over
        if self.over:
            return

        if "hit" in message:
            self.hit = message["hit"]

        self.determine_episode_over()

    def on_cross_start(self, message: Dict[str, Any]) -> None:
        logger.info(f"crossed start line: lap_time {message['lap_time']}")

    def on_race_start(self, message: Dict[str, Any]) -> None:
        logger.debug("race started")

    def on_race_stop(self, message: Dict[str, Any]) -> None:
        logger.debug("race stoped")

    def on_missed_checkpoint(self, message: Dict[str, Any]) -> None:
        logger.info("racer missed checkpoint")
        self.missed_checkpoint = True

    def on_DQ(self, message: Dict[str, Any]) -> None:
        logger.info("racer DQ")
        self.dq = True

    def on_ping(self, message: Dict[str, Any]) -> None:
        """
        no reply needed at this point. Server sends these as a keep alive to make sure clients haven't gone away.
        """
        pass

    def set_episode_over_fn(self, ep_over_fn: Callable[[], bool]):
        """
        allow userd to define their own episode over function
        """
        self.determine_episode_over = types.MethodType(ep_over_fn, self)
        logger.debug("custom ep_over fn set.")

    def determine_episode_over(self):
        # 出界判负:|cte|>max_cte 即终止(reset 后前 5 个 step 给宽限,只覆盖出生瞬间
        # 的异常大 cte;宽限不保护"冲太远"——|cte| 极大同样终止,否则车冲出去后反而
        # 永不判负,在野外空跑到截断)。
        if math.fabs(self.cte) > self.max_cte:
            if self.steps_since_reset < 5 and math.fabs(self.cte) > 2 * self.max_cte:
                pass
            else:
                logger.debug(f"game over: off track cte {self.cte}")
                self.over = True
        elif self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
            self.over = True
        elif self.missed_checkpoint:
            logger.debug("missed checkpoint")
            self.over = True
        elif self.dq:
            logger.debug("disqualified")
            self.over = True

        # sdsandbox RL 终止条件:翻车/卡死(遥测字段驱动,可经 conf 关闭)
        if self.flipped and self.done_on_flip:
            logger.debug("game over: flipped")
            self.over = True
        if self.stuck_time > self.stuck_time_thresh and self.done_on_stuck:
            logger.debug("game over: stuck")
            self.over = True

        # Disable reset
        if os.environ.get("RACE") == "True":
            self.over = False

    def on_scene_selection_ready(self, message: Dict[str, Any]) -> None:
        logger.debug("SceneSelectionReady")
        self.send_get_scene_names()

    def on_car_loaded(self, message: Dict[str, Any]) -> None:
        logger.debug("car loaded")
        self.loaded = True
        # Enable hand brake, so the car doesn't move
        self.send_control(0, 0, 1.0)
        self.on_need_car_config({})

    def on_recv_scene_names(self, message: Dict[str, Any]) -> None:
        if message:
            names = message["scene_names"]
            logger.debug(f"SceneNames: {names}")
            print("loading scene", self.SceneToLoad)
            if self.SceneToLoad in names:
                self.send_load_scene(self.SceneToLoad)
            else:
                raise ValueError(f"Scene name {self.SceneToLoad} not in scene list {names}")

    def send_control(self, steer: float, throttle: float, brake: float = 0.0) -> None:
        """
        Send command to simulator.

        :param steer: desired steering
        :param throttle: desired throttle
        :param brake: whether to activate or not hand brake
            (can be a continuous value)
        """
        if not self.loaded:
            return
        msg = {
            "msg_type": "control",
            "steering": str(steer),
            "throttle": str(throttle),
            "brake": str(brake),
        }
        self.queue_message(msg)

    def send_reset_car(self, seed: Optional[int] = None) -> None:
        # sdsandbox 随机出生:spawn_cfg 带 reset_car 可选字段子集,seed 完全可复现
        msg = {"msg_type": "reset_car"}
        if self.spawn_cfg:
            for key, val in self.spawn_cfg.items():
                msg[key] = val
        if seed is not None:
            msg["seed"] = seed
        self.queue_message(msg)

    def send_get_scene_names(self) -> None:
        msg = {"msg_type": "get_scene_names"}
        self.queue_message(msg)

    def send_load_scene(self, scene_name: str) -> None:
        msg = {"msg_type": "load_scene", "scene_name": scene_name}
        self.queue_message(msg)

    def send_exit_scene(self) -> None:
        msg = {"msg_type": "exit_scene"}
        self.queue_message(msg)

    def send_car_config(
        self,
        body_style: str = "donkey",
        body_rgb: Tuple[int, int, int] = (255, 255, 255),
        car_name: str = "car",
        font_size: int = 100,
    ):
        """
        # body_style = "donkey" | "bare" | "car01" | "f1" | "cybertruck"
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        """
        assert isinstance(body_style, str)
        assert isinstance(body_rgb, list) or isinstance(body_rgb, tuple)
        assert len(body_rgb) == 3
        assert isinstance(car_name, str)
        assert isinstance(font_size, int) or isinstance(font_size, str)

        msg = {
            "msg_type": "car_config",
            "body_style": body_style,
            "body_r": str(body_rgb[0]),
            "body_g": str(body_rgb[1]),
            "body_b": str(body_rgb[2]),
            "car_name": car_name,
            "font_size": str(font_size),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_racer_bio(self, racer_name: str, car_name: str, bio: str, country: str, guid: str) -> None:
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        # guid = "some random string"
        msg = {
            "msg_type": "racer_info",
            "racer_name": racer_name,
            "car_name": car_name,
            "bio": bio,
            "country": country,
            "guid": guid,
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_cam_config(
        self,
        msg_type: str = "cam_config",
        img_w: int = 0,
        img_h: int = 0,
        img_d: int = 0,
        img_enc: Union[str, int] = 0,  # 0 is default value
        fov: int = 0,
        fish_eye_x: float = 0.0,
        fish_eye_y: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        rot_x: float = 0.0,
        rot_y: float = 0.0,
        rot_z: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        """Camera config
        set any field to Zero to get the default camera setting.
        offset_x moves camera left/right
        offset_y moves camera up/down
        offset_z moves camera forward/back
        rot_x will rotate the camera
        with fish_eye_x/y == 0.0 then you get no distortion
        img_enc can be one of JPG|PNG|TGA
        """
        msg = {
            "msg_type": msg_type,
            "fov": str(fov),
            "fish_eye_x": str(fish_eye_x),
            "fish_eye_y": str(fish_eye_y),
            "img_w": str(img_w),
            "img_h": str(img_h),
            "img_d": str(img_d),
            "img_enc": str(img_enc),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
            "rot_y": str(rot_y),
            "rot_z": str(rot_z),
        }
        # 可选遥测推流帧率:>0 时才发(旧模拟器无 fps 字段解析,缺省即兼容)
        if fps > 0:
            msg["fps"] = str(fps)
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_lidar_config(
        self,
        deg_per_sweep_inc: float = 2.0,
        deg_ang_down: float = 0.0,
        deg_ang_delta: float = -1.0,
        num_sweeps_levels: int = 1,
        max_range: float = 50.0,
        noise: float = 0.5,
        offset_x: float = 0.0,
        offset_y: float = 0.5,
        offset_z: float = 0.5,
        rot_x: float = 0.0,
    ):
        """Lidar config
        offset_x moves lidar left/right
        the offset_y moves lidar up/down
        the offset_z moves lidar forward/back
        deg_per_sweep_inc : as the ray sweeps around, how many degrees does it advance per sample (int)
        deg_ang_down : what is the starting angle for the initial sweep compared to the forward vector
        deg_ang_delta : what angle change between sweeps
        num_sweeps_levels : how many complete 360 sweeps (int)
        max_range : what it max distance we will register a hit
        noise : what is the scalar on the perlin noise applied to point position

        Here's some sample settings that similate a more sophisticated lidar:
        msg = '{ "msg_type" : "lidar_config",
        "degPerSweepInc" : "2.0", "degAngDown" : "25", "degAngDelta" : "-1.0",
        "numSweepsLevels" : "25", "maxRange" : "50.0", "noise" : "0.2",
        "offset_x" : "0.0", "offset_y" : "1.0", "offset_z" : "1.0", "rot_x" : "0.0" }'
        And here's some sample settings that similate a simple RpLidar A2 one level horizontal scan.
        msg = '{ "msg_type" : "lidar_config", "degPerSweepInc" : "2.0",
        "degAngDown" : "0.0", "degAngDelta" : "-1.0", "numSweepsLevels" : "1",
        "maxRange" : "50.0", "noise" : "0.4",
        "offset_x" : "0.0", "offset_y" : "0.5", "offset_z" : "0.5", "rot_x" : "0.0" }'
        """
        msg = {
            "msg_type": "lidar_config",
            "degPerSweepInc": str(deg_per_sweep_inc),
            "degAngDown": str(deg_ang_down),
            "degAngDelta": str(deg_ang_delta),
            "numSweepsLevels": str(num_sweeps_levels),
            "maxRange": str(max_range),
            "noise": str(noise),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

        self.lidar_deg_per_sweep_inc = float(deg_per_sweep_inc)
        self.lidar_num_sweep_levels = int(num_sweeps_levels)
        self.lidar_deg_ang_delta = float(deg_ang_delta)

    def process_lidar_packet(self, lidar_info: List[Dict[str, float]]) -> np.ndarray:
        point_per_sweep = int(360 / self.lidar_deg_per_sweep_inc)
        points_num = round(abs(self.lidar_num_sweep_levels * point_per_sweep))
        reconstructed_lidar_info = [-1 for _ in range(points_num)]  # we chose -1 to be the "None" value

        if lidar_info is not None:
            for point in lidar_info:
                rx = point["rx"]
                ry = point["ry"]
                d = point["d"]

                x_index = round(abs(rx / self.lidar_deg_per_sweep_inc))
                y_index = round(abs(ry / self.lidar_deg_ang_delta))

                reconstructed_lidar_info[point_per_sweep * y_index + x_index] = d

        return np.array(reconstructed_lidar_info)

    def blocking_send(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"blocking send \n {msg}")
        self.client.send_now(msg)

    def queue_message(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"sending \n {msg}")
        self.client.queue_message(msg)
