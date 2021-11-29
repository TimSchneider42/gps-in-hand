from abc import ABC, abstractmethod
from typing import Dict, Callable
from typing import Optional, TYPE_CHECKING, Tuple, Iterable

from allegro_pybullet.util import ReadOnlyDict

from allegro_pybullet.simulation_object import SimulationObject

if TYPE_CHECKING:
    from allegro_pybullet.physics_client import PhysicsClient


class SimulationBody(ABC):
    def __init__(self):
        self.__body_unique_id = None  # type: Optional[int]
        self.__physics_client = None  # type: Optional[PhysicsClient]
        self.__simulation_objects = None  # type: Optional[Dict[str, SimulationObject]]

    def _on_reset(self):
        pass

    def call(self, func: Callable, *args, **kwargs):
        assert self.is_setup, "This body has not yet been setup."
        return self.__physics_client.call(func, self.__body_unique_id, *args, **kwargs)

    @abstractmethod
    def _setup(self) -> Tuple[int, Iterable[SimulationObject]]:
        """
        Sets up this body in the simulation and returns the body ID and all simulation objects.
        :return:
        """
        pass

    @property
    def body_unique_id(self) -> Optional[int]:
        """
        Body ID of this body in the simulation.
        :return:
        """
        return self.__body_unique_id

    @property
    def is_setup(self) -> bool:
        return self.__body_unique_id is not None

    @property
    def physics_client(self) -> Optional["PhysicsClient"]:
        return self.__physics_client

    @property
    def simulation_objects(self) -> Optional[Dict[str, SimulationObject]]:
        return self.__simulation_objects

    # ==================================================================================================================
    # Interface to communicate with PhysicsClient. Do not call these functions directly.

    def _register_physics_client(self, physics_client: "PhysicsClient"):
        """
        Sets up this body in the simulation.
        :return:
        """
        assert not self.is_setup, "This body has already been setup."
        assert self in physics_client.bodies, "This body is not part of the given simulation."
        self.__physics_client = physics_client
        self.__body_unique_id, simulation_objects = self._setup()
        self.__simulation_objects = ReadOnlyDict({so.name: so for so in simulation_objects})

    def _unregister_physics_client(self):
        assert self.is_setup, "This body has not yet been setup."
        self._on_reset()
        self.__simulation_objects = None
        self.__body_unique_id = None

    def _observe(self):
        SimulationObject._observe_all(self, self.simulation_objects.values())

    def _act(self):
        SimulationObject._act_all(self, self.simulation_objects.values())

    def _set_initial_state(self):
        SimulationObject._set_initial_state_all(self, self.simulation_objects.values())
