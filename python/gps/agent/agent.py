""" This file defines the base agent class. """
from abc import abstractmethod
from threading import Lock
from typing import Dict, Any, Optional, List

import numpy as np

from gps.agent.noise_generator import NoiseGenerator
from gps.controller import Controller
from gps.utility.abortable_worker import AbortableWorker
from gps.utility.labeled_data_packer import LabeledDataPacker
from gps.sample import Sample, SampleCollator


class Agent(AbortableWorker):
    """
    Agent superclass. The agent interacts with the environment to
    collect samples.
    """

    def __init__(self, time_steps: int, condition_count: int, action_dimensions: int,
                 state_packer: LabeledDataPacker, observation_packer: Optional[LabeledDataPacker] = None,
                 tracking_point_labels: List[Any] = None):
        """

        :param time_steps:              Number of time steps for each sample.
        :param condition_count:         Number of conditions this agent can be setup to.
        :param action_dimensions:       Dimensions of the action vector.
        :param state_packer:            Data packer for the state.
        :param observation_packer:      Data packer for the observation (Default: same as state packer).
        :param tracking_point_labels:   Labels of the tracking points (Default: none). This information is used
                                        by the GUI to visualize trajectories.
        """
        AbortableWorker.__init__(self)
        self._condition_count = condition_count
        self._time_steps = time_steps
        self._action_dimensions = action_dimensions
        self._state_packer = state_packer
        if observation_packer is None:
            self._observation_packer = state_packer
        else:
            self._observation_packer = observation_packer
        self._is_sampling = False
        self._initialized = False
        self._terminated = False
        self._terminating = False
        self._state_lock = Lock()
        self._tracking_point_labels = tracking_point_labels if tracking_point_labels is not None else []

    def _run(self, controller: Controller, condition: int,
             controller_noise_generator: Optional[NoiseGenerator] = None) -> Optional[Sample]:
        """
        Draw a sample from the environment, using the specified controller and under the specified condition.
        :param controller:                      The controller used for the action generation.
        :param condition:                   Which condition setup to run.
        :param controller_noise_generator:  Generator for the noise that will be scaled and applied to the control
                                            vector (Default: no noise).
        :return: The Sample.
        """
        self._check_state()
        try:
            self._is_sampling = True
            if condition not in range(self.condition_count):
                raise ValueError(
                    "Unknown condition {0}, expected [0-{1}]".format(condition, self.condition_count - 1))

            if controller_noise_generator is not None:
                noise = controller_noise_generator.generate_noise(self.time_steps, self.action_dimensions)
            else:
                noise = np.zeros((self._time_steps, self.action_dimensions))

            full_state = self._reset(condition)
            sample_collator = SampleCollator(full_state, self.time_steps, self.action_dimensions, self.state_packer,
                                             self.observation_packer)
            controller.prepare_sampling()
            for t in range(self.time_steps - 1):
                if self.aborting or self._terminating:
                    self._on_sample_aborted()
                    return None
                state = self.state_packer.pack(full_state)
                observation = self.observation_packer.pack(full_state)
                action = controller.act(t, state, observation, noise[t, :])
                full_state = self._do_step(action, t)
                sample_collator.add(action, full_state)
            controller.sampling_done()
            self._on_sample_complete()
            return sample_collator.finalize()
        finally:
            self._is_sampling = False

    @abstractmethod
    def _reset(self, condition: int) -> Dict[Any, np.array]:
        """
        Reset this agent to its initial state
        :param condition: Which condition setup to run.
        :return: The initial state.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _do_step(self, action: np.array, time_step: int) -> Dict[Any, np.array]:
        """
        Transfer system to the next state using the specified action.
        :param action: The action to apply on the system.
        :param time_step: Current iteration number
        :return: The new state.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def _on_sample_complete(self):
        """
        Can be implemented in subclass to handle completion of the current sample.
        :return:
        """
        pass

    def _on_sample_aborted(self):
        """
        Can be implemented in subclass to handle abortion of the current sample.
        :return:
        """
        pass

    def _on_initialize(self):
        """
        Can be implemented in subclass. This will be called once when the agent is initialized.
        :return:
        """
        pass

    def _on_terminate(self):
        """
        Can be implemented in subclass. This will be called once when the agent terminates.
        :return:
        """
        pass

    def initialize(self):
        with self._state_lock:
            if self._initialized:
                raise ValueError("This agent has already been initialized.")
            self._on_initialize()
            self._initialized = True

    def terminate(self):
        with self._state_lock:
            self._check_state()
            self._terminating = True
            self.abort()
            self._on_terminate()
            self._terminated = True
            self._terminating = False

    def debug_command(self, cmd: str):
        pass

    def _check_state(self):
        if not self._initialized:
            raise ValueError("This agent has not yet been initialized.")
        if self._terminating:
            raise ValueError("This agent is terminating.")
        if self._terminated:
            raise ValueError("This agent has already terminated.")

    @property
    def state_packer(self) -> LabeledDataPacker:
        return self._state_packer

    @property
    def observation_packer(self) -> LabeledDataPacker:
        return self._observation_packer

    @property
    def action_dimensions(self) -> int:
        return self._action_dimensions

    @property
    def state_dimensions(self) -> int:
        return self._state_packer.dimensions

    @property
    def observation_dimensions(self) -> int:
        return self._observation_packer.dimensions

    @property
    def time_steps(self) -> int:
        return self._time_steps

    @property
    def condition_count(self) -> int:
        return self._condition_count

    @property
    def is_sampling(self) -> bool:
        return self._is_sampling

    @property
    def tracking_point_labels(self) -> List[Any]:
        return self._tracking_point_labels
