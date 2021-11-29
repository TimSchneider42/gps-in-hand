from typing import Optional, Dict, Any

import numpy as np

from gps.utility.labeled_data_packer import LabeledDataPacker


class Sample:
    """
    Class that stores a single trajectory.
    """

    def __init__(self, full_state: Dict[Any, np.ndarray], actions: np.ndarray, state_packer: LabeledDataPacker,
                 observation_packer: Optional[LabeledDataPacker] = None):
        """

        :param state_packer: DataPacker that packs together the state vector
        :param action_dimensions: Dimensions of the action vector
        :param time_steps: Total number of time steps for this sample
        :param observation_packer: DataPacker that packs together the observation vector. Default: same as state_packer.
        :param extra_state_dimensions: Dictionary that stores the dimensions of each sensor that does not appear in
                                        either state_packer or observation_packer.
        """
        self._time_steps = actions.shape[0] + 1
        self._action_dimensions = actions.shape[1]
        self._state_packer = state_packer
        self._observation_packer = observation_packer if observation_packer is not None else state_packer
        self._current_time_step = 0

        self._states = state_packer.pack(full_state)
        self._observations = observation_packer.pack(full_state)
        self._actions = actions

        self._full_state = full_state

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
    def states(self) -> np.ndarray:
        """
        Returns all state vectors (T x dX).
        :return:
        """
        return self._states

    @property
    def actions(self) -> np.ndarray:
        """
        Returns all action vectors (T x dU).
        :return:
        """
        return self._actions

    @property
    def observations(self) -> np.ndarray:
        """
        Returns all observation vectors.
        :return:
        """
        return self._observations

    @property
    def full_state(self) -> Dict[Any, np.ndarray]:
        """
        Returns a dictionary containing all recorded sensor values for this sample.
        :return:
        """
        return self._full_state
