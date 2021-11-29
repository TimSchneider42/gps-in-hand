from typing import Optional, Dict, Any

import numpy as np

from gps.sample import Sample
from gps.utility.labeled_data_packer import LabeledDataPacker


class SampleCollator:
    """
    Class that collates data of a single trajectory.
    """

    def __init__(self, initial_state: Dict[Any, np.ndarray], time_steps: int, action_dimensions: int,
                 state_packer: LabeledDataPacker, observation_packer: Optional[LabeledDataPacker] = None):
        """

        :param initial_state:           Initial state of the trajectory
        :param time_steps:              Total number of time steps for this sample
        :param action_dimensions:       Dimensions of the action vector
        :param state_packer:            DataPacker that packs together the state vector
        :param observation_packer:      DataPacker that packs together the observation vector. Default: same as
                                        state_packer.
        """
        self._time_steps = time_steps
        self._action_dimensions = action_dimensions
        self._state_packer = state_packer
        self._observation_packer = observation_packer if observation_packer is not None else state_packer
        self._current_time_step = 0

        self._states = None
        self._observations = None
        self._actions = np.empty((self.time_steps - 1, self.action_dimensions))

        self._full_state = {s: np.zeros((self.time_steps, len(v))) for s, v in initial_state.items()}
        for s, v in initial_state.items():
            self._full_state[s][0, :] = v
        self._current_time_step = 1

    def add(self, action: np.ndarray, state: Dict[Any, np.ndarray]):
        """
        Adds an action and the resulting state
        :param action: Action required to go from previous state to the specified new one
        :param state: The state after applying the specified action
        """
        if self._current_time_step >= self.time_steps:
            raise ValueError("The sample is already complete.")
        if self._current_time_step == 0:
            raise ValueError("Need to set initial state first.")
        for s in self._full_state:
            self._full_state[s][self._current_time_step, :] = state[s]
        self._actions[self._current_time_step - 1, :] = action
        self._current_time_step += 1

    def finalize(self) -> Sample:
        if not self.complete:
            raise ValueError("The sample needs to be complete before it can be finalized.")
        return Sample(full_state=self._full_state, actions=self._actions, state_packer=self._state_packer,
                      observation_packer=self._observation_packer)

    @property
    def time_steps(self) -> int:
        """
        Total number of time steps.
        :return:
        """
        return self._time_steps

    @property
    def state_packer(self) -> LabeledDataPacker:
        """
        Data packer for the state.
        :return:
        """
        return self._state_packer

    @property
    def state_dimensions(self) -> int:
        """
        Dimensions of the state.
        :return:
        """
        return self.state_packer.dimensions

    @property
    def observation_packer(self) -> LabeledDataPacker:
        """
        Data packer for the observation
        :return:
        """
        return self._observation_packer

    @property
    def observation_dimensions(self) -> int:
        """
        Dimensions of the observation
        :return:
        """
        return self.observation_packer.dimensions

    @property
    def action_dimensions(self) -> int:
        """
        Dimensions of the action vector
        :return:
        """
        return self._action_dimensions

    @property
    def complete(self) -> bool:
        """
        Returns true if this sample is complete.
        :return:
        """
        return self._current_time_step == self.time_steps

    @property
    def full_state(self) -> Dict[Any, np.ndarray]:
        """
        Returns a dictionary containing all recorded sensor values for this sample.
        :return:
        """
        return self._full_state
