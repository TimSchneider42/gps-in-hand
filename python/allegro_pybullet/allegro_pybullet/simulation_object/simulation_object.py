from abc import ABC
from typing import List, TYPE_CHECKING, Iterable

import itertools
from typing import Type

if TYPE_CHECKING:
    from allegro_pybullet.simulation_body import SimulationBody


class SimulationObject(ABC):
    """
    Represents an object in the simulation.
    """

    def __init__(self, name: str):
        """
        :param name:    Name of the object in the simulator.
        """
        self.__name = name

    @property
    def name(self) -> str:
        """
        Name of the object in the simulator.
        :return:
        """
        return self.__name

    @classmethod
    def _observe_all(cls, body: "SimulationBody", simulation_objects: Iterable["SimulationObject"]):
        """
        A call to this function will update the observed variables of all SimulationObject instances in the given list.
        This means that for each type of SimulationObject in the list, the _observe function is called once with all
        instances of that type in the list.
        :param simulation_objects:  SimulationObject instances to update observation on.
        :return:
        """
        # Group by type and iterate
        for t, sim_objs in itertools.groupby(simulation_objects, type):  # type: Type[SimulationObject]
            t._observe(body, list(sim_objs))

    @classmethod
    def _observe(cls, body: "SimulationBody", simulation_objects: List["SimulationObject"]):
        """
        Updates the observation of the given list of SimulationObject instances. This function is supposed to be
        overridden by a subclass if needed.
        :param body:     Body of all given simulation_objects.
        :param simulation_objects:  SimulationObjects to update (expected to be of the type of the overriding subclass).
        :return:
        """
        pass

    @classmethod
    def _act_all(cls, body: "SimulationBody", simulation_objects: Iterable["SimulationObject"]):
        """
        A call to this function will make all SimulationObject instances in the given list act in the simulation.
        This means that for each type of SimulationObject in the list, the _act function is called once with all
        instances of that type in the list.
        :param simulation_objects:  SimulationObject instances to act.
        :return:
        """
        # Group by type and iterate
        for t, sim_objs in itertools.groupby(simulation_objects, type):  # type: Type[SimulationObject]
            t._act(body, list(sim_objs))

    @classmethod
    def _act(cls, body: "SimulationBody", simulation_objects: List["SimulationObject"]):
        """
        Makes all SimulationObject instances in the given list act. This function is supposed to be overridden by a
        subclass if needed.
        :param body:     Body of all given simulation_objects.
        :param simulation_objects:  SimulationObjects to act (expected to be of the type of the overriding subclass).
        :return:
        """
        pass

    @classmethod
    def _set_initial_state_all(cls, body: "SimulationBody", simulation_objects: Iterable["SimulationObject"]):
        """
        A call to this function will make all SimulationObject instances in the given list set their initial states in
        the simulation. This means that for each type of SimulationObject in the list, the _set_initial_state function
        is called once with all instances of that type in the list.
        :param simulation_objects:  SimulationObject instances to set initial state.
        :return:
        """
        # Group by type and iterate
        for t, sim_objs in itertools.groupby(simulation_objects, type):  # type: Type[SimulationObject]
            t._set_initial_state(body, list(sim_objs))

    @classmethod
    def _set_initial_state(cls, body: "SimulationBody", simulation_objects: List["SimulationObject"]):
        """
        Makes all SimulationObject instances in the given list set their initial state in the simulation. This function
        is supposed to be overridden by a subclass if needed.
        :param body:     Body of all given simulation_objects.
        :param simulation_objects:  SimulationObjects to set initial state (expected to be of the type of the overriding
                                    subclass).
        :return:
        """
        pass

    def __repr__(self) -> str:
        return self.name
