import pybullet
from typing import List
from typing import Optional, Tuple

import numpy as np

from allegro_pybullet.simulation_body import SimulationBody


class PhysicsClient:
    def __init__(self):
        self.__physics_client_id = None  # type. Optional[int
        self.__bodies = []  # type: List[SimulationBody]
        self.__real_time_simulation = False
        self.__gravity = np.zeros(3)

    def __connect(self, *args, **kwargs):
        assert not self.is_connected, "This client has already been connected."
        self.__physics_client_id = pybullet.connect(*args, **kwargs)

    def connect_direct(self):
        self.__connect(pybullet.DIRECT)

    def connect_gui(self, options: str = ""):
        self.__connect(pybullet.GUI, options=options)

    # Currently not supported as this won't work with multiple clients
    """
    def connect_udp(self, hostname: str, port: Optional[int] = None):
        if port is not None:
            self.connect(pybullet.UDP, hostname, port)
        else:
            self.connect(pybullet.UDP, hostname)

    def connect_tcp(self, hostname: str, port: Optional[int] = None):
        if port is not None:
            self.connect(pybullet.TCP, hostname, port)
        else:
            self.connect(pybullet.TCP, hostname)

    def connect_shared_memory(self, key: Optional[int] = None):
        if key is not None:
            self.connect(pybullet.SHARED_MEMORY, key=key)
        else:
            self.connect(pybullet.SHARED_MEMORY)
    """

    def disconnect(self):
        self.reset_simulation()
        self.call(pybullet.disconnect)
        self.__physics_client_id = None

    def reset_simulation(self):
        for body in self.__bodies:
            body._unregister_physics_client()
        self.__bodies.clear()
        self.call(pybullet.resetSimulation)

    def call(self, func, *args, **kwargs):
        assert self.is_connected, "This client has not yet been connected."
        assert "physicsClientId" not in kwargs, \
            "The argument physicsClientId will be filled in by the PhysicsClient instance and thus needs to be left " \
            "empty."
        kwargs["physicsClientId"] = self.physics_client_id
        return func(*args, **kwargs)

    def add_body(self, simulation_body: SimulationBody):
        try:
            self.__bodies.append(simulation_body)
            simulation_body._register_physics_client(self)
            simulation_body._set_initial_state()
            simulation_body._observe()
        except Exception:
            self.__bodies.remove(simulation_body)
            raise

    def reset_to_initial_state(self):
        for body in self.bodies:
            body._set_initial_state()
            body._observe()

    def step_simulation(self):
        for body in self.bodies:
            body._act()
        self.call(pybullet.stepSimulation)
        for body in self.bodies:
            body._observe()

    def reset_debug_visualizer_camera(self, camera_distance: float, camera_yaw: float, camera_pitch: float,
                                      camera_target_position: np.ndarray):
        self.call(pybullet.resetDebugVisualizerCamera, camera_distance, camera_yaw, camera_pitch,
                  camera_target_position)

    def configure_debug_visualizer(self, flag: int, enable: bool):
        self.call(pybullet.configureDebugVisualizer, flag, enable)

    def set_additional_search_path(self, path: str):
        self.call(pybullet.setAdditionalSearchPath, path)

    @property
    def gravity(self):
        return self.__gravity

    @gravity.setter
    def gravity(self, value: np.ndarray):
        assert value.shape == (3,)
        self.__gravity = value
        self.call(pybullet.setGravity, *value)

    @property
    def real_time_simulation(self) -> bool:
        return self.__real_time_simulation

    @real_time_simulation.setter
    def real_time_simulation(self, value: bool):
        self.__real_time_simulation = value
        self.call(pybullet.setRealTimeSimulation, value)

    @property
    def time_step(self) -> float:
        return self.call(pybullet.getPhysicsEngineParameters)["fixedTimeStep"]

    @time_step.setter
    def time_step(self, value: float):
        self.call(pybullet.setTimeStep, value)

    @property
    def physics_client_id(self) -> Optional[int]:
        return self.__physics_client_id

    @property
    def is_connected(self) -> bool:
        return self.__physics_client_id is not None

    @property
    def bodies(self) -> Tuple[SimulationBody, ...]:
        return tuple(self.__bodies)
