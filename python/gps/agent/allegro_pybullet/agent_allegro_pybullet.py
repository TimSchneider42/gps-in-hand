import glob
import shutil
import subprocess
from enum import Enum
from tempfile import NamedTemporaryFile, mkdtemp
from time import time, sleep
from typing import List, Any, Dict, Optional, Iterable, Tuple

import logging

import math
import numpy as np
import os

import pybullet

import pybullet_data
from PIL import Image, ImageFont, ImageDraw

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body import URDFBody
from allegro_pybullet.simulation_object import JointControlMode
from gps.agent import Agent
from gps.agent.allegro_pybullet.environment_plugin import EnvironmentPlugin
from gps.agent.allegro_pybullet.initialization_sequence import InitializationSequence
from gps.utility.labeled_data_packer import LabeledDataPacker
from gps.utility.timer import Timer
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand, AllegroFingerType, AllegroFingerLinkType, \
    AllegroFingerJointType

LOGGER = logging.getLogger(__name__)


class Property(Enum):
    JOINT_ANGLES = 0
    JOINT_VELOCITIES = 1
    TIP_POSITION = 2
    TACTILE_FORCES = 3
    SUMMED_TACTILE_FORCES = 4
    # Includes unused joints on USED FINGERS(!)
    JOINT_ANGLES_INCLUDE_UNUSED = 5


class AgentAllegroPybullet(Agent):
    """
    Implementation of the Agent interface for Darias.
    """

    def __init__(self, time_steps: int, time_step_seconds: float,
                 used_fingers: List[AllegroFingerType], initialization_sequences: List[InitializationSequence],
                 state_properties: Optional[Iterable[Property]] = None,
                 observation_properties: Optional[Iterable[Property]] = None,
                 environment_plugins: Optional[Iterable[EnvironmentPlugin]] = None,
                 real_time_factor: Optional[float] = None,
                 pre_start_pause: float = 0.0, show_gui: bool = False, gui_options: str = "", sub_steps: int = 0,
                 used_joints: Optional[List[AllegroFingerJointType]] = None, recording_directory: Optional[str] = None,
                 recording_width: int = 1000, recording_height: int = 1000,
                 max_joint_torque: Optional[float] = None, enable_gravity: bool = False,
                 hand_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 ground_plane_collision: bool = False):
        self._physics_client = PhysicsClient()

        self.__pre_start_pause = pre_start_pause
        self._real_time_factor = real_time_factor
        self._time_step_seconds = time_step_seconds
        self._used_finger_joints = AllegroFingerJointType if used_joints is None else used_joints
        self._show_gui = show_gui
        self._sub_steps = sub_steps
        self._gui_options = gui_options
        self._recording_directory = recording_directory
        self._recording_width = recording_width
        self._recording_height = recording_height
        self._recording_view_mat = None
        self._recording_proj_mat = None
        self._logging_unique_id: Optional[int] = None
        self._max_joint_torque = np.inf if max_joint_torque is None else max_joint_torque
        environment_plugins = [] if environment_plugins is None else environment_plugins
        self._environment_plugins = environment_plugins
        self._ground_plane_collision = ground_plane_collision

        self._used_fingers = used_fingers
        if hand_pose is None:
            hand_pose = (np.array([0.0, 0.0, 0.1]),
                         np.array(pybullet.getQuaternionFromEuler([math.pi / 2, 0, math.pi])))
        self._hand = AllegroRightHand(base_position=hand_pose[0], base_orientation=hand_pose[1])
        self._enable_gravity = enable_gravity

        finger_labels = [((jp, t), len(self._used_finger_joints)) for t in self._used_fingers for jp in
                         [Property.JOINT_ANGLES, Property.JOINT_VELOCITIES]] + \
                        [((Property.JOINT_ANGLES_INCLUDE_UNUSED, t), 4) for t in self._used_fingers] + \
                        [((Property.TACTILE_FORCES, t), 23) for t in self._used_fingers] + \
                        [((Property.TIP_POSITION, t), 3) for t in self._used_fingers] + \
                        [((Property.SUMMED_TACTILE_FORCES, t), 1) for t in self._used_fingers]

        # By default, use all labels
        observation_properties = Property if observation_properties is None else observation_properties
        state_properties = Property if state_properties is None else state_properties

        state_labels = sum([e.state_labels for e in environment_plugins],
                           [l for l in finger_labels if l[0][0] in state_properties])

        observation_labels = sum([e.observation_labels for e in environment_plugins],
                                 [l for l in finger_labels if l[0][0] in observation_properties])

        state_packer = LabeledDataPacker(state_labels)
        observation_packer = LabeledDataPacker(observation_labels)
        action_dimensions = len(used_joints) * len(used_fingers)
        super(AgentAllegroPybullet, self).__init__(
            time_steps, len(initialization_sequences), action_dimensions,
            state_packer=state_packer,
            observation_packer=observation_packer,
            tracking_point_labels=[(Property.TIP_POSITION, t) for t in self._used_fingers])

        self._initialization_sequences = initialization_sequences

        self._next_step: Optional[float] = None

        # List for collecting timing information
        # pre step delay; step delay;
        self._timing_seconds = np.zeros((time_steps - 1, 2))
        self._timer = Timer()

        self._tmp_dir: Optional[str] = None
        self._current_frame_no = 0
        real_time_factor = 1.0 if self._real_time_factor is None else self._real_time_factor
        self._fps = real_time_factor / self._time_step_seconds
        self._current_condition = None
        data_dir = os.path.join(os.path.dirname(__file__), "allegro_pybullet_data")
        font_size = int(min(self._recording_width, self._recording_height) * 0.06)
        self._font = ImageFont.truetype(os.path.join(data_dir, "DejaVuSans.ttf"), font_size)

    def _on_initialize(self):
        LOGGER.info("Initializing pybullet...")
        if self._show_gui:
            self._physics_client.connect_gui(options=self._gui_options)
        else:
            self._physics_client.connect_direct()
        self._physics_client.time_step = self._time_step_seconds
        self._physics_client.call(pybullet.setPhysicsEngineParameter, numSubSteps=self._sub_steps)

        LOGGER.info("Loading URDF models...")
        if self._ground_plane_collision:
            # Add a plane from the standard pybullet data to the simulation.
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            plane = URDFBody("plane.urdf")
        else:
            plane = URDFBody(os.path.join(os.path.dirname(__file__), "allegro_pybullet_data/plane.urdf"))
        self._physics_client.add_body(plane)

        self._physics_client.add_body(self._hand)
        self._physics_client.reset_debug_visualizer_camera(0.7, -15, 250, self._hand.base_link.initial_position)
        self._physics_client.configure_debug_visualizer(pybullet.COV_ENABLE_MOUSE_PICKING, False)

        # Set lateral friction of hand joints
        for ft in AllegroFingerType:
            self._hand.fingers[ft].links[AllegroFingerLinkType.TIP].lateral_friction = 0.8

        LOGGER.info("Loading plugins...")
        for e in self._environment_plugins:
            e.on_initialize(self._physics_client, self._hand)

        for s in self._initialization_sequences:
            s.on_initialize(self._physics_client, self._hand)

        LOGGER.info("pybullet initialized.")

    def debug_command(self, cmd: str):
        yaw, pitch, dist, target = self._physics_client.call(pybullet.getDebugVisualizerCamera)[8:12]
        target = np.array(target)
        c, *args = cmd.split(" ")
        if c in ["d", "y", "p", "tx", "ty", "tz"]:
            if c == "d":
                dist = float(args[0])
            elif c == "y":
                yaw = float(args[0])
            elif c == "p":
                pitch = float(args[0])
            elif c == "tx":
                target[0] = float(args[0])
            elif c == "ty":
                target[1] = float(args[0])
            elif c == "tz":
                target[2] = float(args[0])
            self._physics_client.reset_debug_visualizer_camera(dist, yaw, pitch, target)
        elif c == "s":
            a = {"d": dist, "y": yaw, "p": pitch, "r": 0, "tx": target[0], "ty": target[1], "tz": target[2], "w": 1000,
                 "h": 1000, "z": 0.03, "pn": 0.05, "pf": 200.0, "la": 0.5, "f": "snapshot.jpg"}
            for n, v in [args[2 * i:2 * (i + 1)] for i in range(len(args) // 2)]:
                a[n] = eval(v)
            mv = self._physics_client.call(pybullet.computeViewMatrixFromYawPitchRoll,
                                           [a["tx"], a["ty"], a["tz"]], a["d"], a["y"], a["p"], a["r"], 2)
            mp = self._physics_client.call(pybullet.computeProjectionMatrix, a["z"], -a["z"], a["z"], -a["z"], a["pn"],
                                           a["pf"])
            self._take_snapshot(width=a["w"], height=a["h"], view_mat=mv, projection_mat=mp, file_name=a["f"])
        elif c == "g":
            time_steps = int(args[0])
            next_time_step = None
            for t in range(time_steps):
                if self._real_time_factor is not None:
                    t = time()
                    if next_time_step is not None:
                        sleep(max(0.0, next_time_step - t))
                    else:
                        next_time_step = t
                    next_time_step = max(next_time_step + self._physics_client.time_step / self._real_time_factor, t)
                self._physics_client.step_simulation()
        elif c == "c":
            info = self._physics_client.call(pybullet.getDebugVisualizerCamera)
            print(f"d {info[10]} y {info[8]} p {info[9]} t {info[11]}")

    def _on_terminate(self):
        for e in self._environment_plugins:
            e.on_terminate()
        self._physics_client.disconnect()

    def _extract_used_joints(self, state: Iterable[float]) -> np.ndarray:
        used_indices = [jt.value for jt in self._used_finger_joints]
        return np.take(state, used_indices)

    def _take_snapshot(self, time_step: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None,
                       view_mat: Optional[Iterable[Iterable[float]]] = None,
                       projection_mat: Optional[Iterable[Iterable[float]]] = None,
                       file_name: Optional[str] = None):
        if view_mat is None:
            view_mat = self._recording_view_mat
        if projection_mat is None:
            projection_mat = self._recording_proj_mat
        if width is None:
            width = self._recording_width
        if height is None:
            height = self._recording_height
        wh = max(width, height)
        w, h, px, *_ = self._physics_client.call(
            pybullet.getCameraImage, wh * 2, wh * 2, view_mat, projection_mat, shadow=1,
            lightDirection=[1.0, -1.0, 1.0], lightDistance=2.0, lightColor=[1.0, 1.0, 1.0],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        img = Image.fromarray(px).convert("RGB").resize(
            (wh, wh), Image.ANTIALIAS).crop(
            [(wh - width) / 2, (wh - height) / 2,
             (wh + width) / 2, (wh + height) / 2])
        if time_step is not None:
            draw = ImageDraw.Draw(img)
            time_text = "{:0.2f}s".format(time_step * self._time_step_seconds)
            tw, th = draw.textsize(time_text, font=self._font)
            draw.text((width - 20 - tw, 20), time_text, fill=(0, 0, 0), font=self._font)
        if file_name is None:
            file_name = os.path.join(self._tmp_dir, f"{self._current_frame_no:04d}.jpg")
            self._current_frame_no += 1
        img.save(file_name, "JPEG")

    def _reset(self, condition: int) -> Dict[Any, np.ndarray]:
        self._current_condition = condition
        self._physics_client.gravity = np.zeros(3)

        for e in self._environment_plugins:
            e.reset(condition)

        self._initialization_sequences[condition].run(self._physics_client, self._hand)

        # Set all used joints to torque control mode
        for t in self._used_fingers:
            for jt in self._used_finger_joints:
                joint = self._hand.fingers[t].joints[jt]
                joint.set_control_mode(JointControlMode.TORQUE_CONTROL_HOLD)
                joint.torque_force = 0

        self._next_step = None
        if self._recording_directory is not None:
            self._tmp_dir = mkdtemp()
            self._current_frame_no = 0
            yaw, pitch, dist, target = self._physics_client.call(pybullet.getDebugVisualizerCamera)[8:12]
            z = 0.03
            self._recording_view_mat = self._physics_client.call(
                pybullet.computeViewMatrixFromYawPitchRoll, target, dist, yaw, pitch, 0, 2)
            self._recording_proj_mat = self._physics_client.call(
                pybullet.computeProjectionMatrix, z, -z, z, -z, 0.05, 200.0)
            for i in range(int(self._fps)):
                self._take_snapshot(0)

        sleep(self.__pre_start_pause)
        if self._enable_gravity:
            self._physics_client.gravity = (0, 0, -9.81)
        for e in self._environment_plugins:
            e.on_start()
        return self._get_state()

    def _do_step(self, action: np.ndarray, time_step: int) -> Dict[Any, np.ndarray]:
        uj = len(self._used_finger_joints)
        for i, ft in enumerate(self._used_fingers):
            for j, jt in enumerate(sorted(self._used_finger_joints, key=lambda jt: jt.value)):
                self._hand.fingers[ft].joints[jt].torque_force = min(action[i * uj + j], self._max_joint_torque)
        self._timing_seconds[time_step, 0] = self._timer.round()
        if self._real_time_factor is not None:
            t = time()
            if self._next_step is not None:
                sleep(max(0.0, self._next_step - t))
            else:
                self._next_step = t
            self._next_step = max(self._next_step + self._physics_client.time_step / self._real_time_factor, t)
        self._physics_client.step_simulation()
        self._timing_seconds[time_step, 1] = self._timer.round()
        if self._recording_directory is not None:
            self._take_snapshot(time_step)
        return self._get_state()

    def _on_sample_complete(self):
        for e in self._environment_plugins:
            e.on_sample_complete()
        # Stop logging
        if self._recording_directory is not None:
            for i in range(int(self._fps)):
                self._take_snapshot(self.time_steps)
            # Convert picture to video
            if not os.path.exists(self._recording_directory):
                os.makedirs(self._recording_directory)
            video_path = os.path.join(self._recording_directory, f"log_c{self._current_condition:02d}.avi")
            index = 0
            while os.path.exists(video_path):
                index += 1
                video_path = os.path.join(self._recording_directory,
                                          f"log_c{self._current_condition:02d}_{index:02d}.avi")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-r", str(self._fps),
                "-f", "image2",
                "-s", f"{self._recording_width}x{self._recording_height}",
                "-i", os.path.join(self._tmp_dir, "%04d.jpg"),
                "-vcodec", "libx264",
                "-crf", "15",
                "-pix_fmt", "yuv420p",
                f"{video_path}"
            ], check=True)
            files = list(sorted(glob.glob(os.path.join(self._tmp_dir, "*.jpg"))))
            shutil.move(files[0],
                        os.path.join(self._recording_directory, "log_c{self._current_condition:02d}_init.jpg"))
            shutil.move(files[-1],
                        os.path.join(self._recording_directory, "log_c{self._current_condition:02d}_final.jpg"))
            shutil.rmtree(self._tmp_dir)
        # Evaluate timing
        policy_computation_delays = self._timing_seconds[1:, 0]
        step_delays = self._timing_seconds[:, 1]
        total_time = np.sum(self._timing_seconds)
        simulated_time = self._physics_client.time_step * self.time_steps
        LOGGER.debug(
            """Timing:
                    Policy comp.: avg: {:0.6f}s, min: {:0.6f}s ({:4d}), max: {:0.6f}s ({:4d})
                    Sim. step   : avg: {:0.6f}s, min: {:0.6f}s ({:4d}), max: {:0.6f}s ({:4d})
               Took {:0.6f}s to simulate {:0.6f}s.
            """.format(*[f(t) for t in [policy_computation_delays, step_delays] for f in
                         [np.average, np.min, np.argmin, np.max, np.argmax]], total_time, simulated_time))

    def _get_state(self) -> Dict[Any, np.ndarray]:
        state = {}
        f = self._hand.fingers
        state.update({(Property.JOINT_ANGLES, t): self._extract_used_joints(f[t].angles) for t in self._used_fingers})
        state.update({(Property.JOINT_ANGLES_INCLUDE_UNUSED, t): np.array(f[t].angles) for t in self._used_fingers})
        state.update({(Property.JOINT_VELOCITIES, t): self._extract_used_joints(f[t].angular_velocities) for t in
                      self._used_fingers})
        state.update({(Property.TIP_POSITION, t): f[t].links[AllegroFingerLinkType.TIP].observed_position for t in
                      self._used_fingers})
        state.update({(Property.TACTILE_FORCES, t): f[t].tactile_sensor.tactel_forces for t in self._used_fingers})
        state.update(
            {(Property.SUMMED_TACTILE_FORCES, t): np.array([np.sum(f[t].tactile_sensor.tactel_forces)]) for t in
             self._used_fingers})
        for e in self._environment_plugins:
            state.update(e.get_state())
        return state

    @property
    def tracking_point_labels(self):
        return super(AgentAllegroPybullet, self).tracking_point_labels + [l for e in self._environment_plugins for l in
                                                                          e.tracking_point_labels]
