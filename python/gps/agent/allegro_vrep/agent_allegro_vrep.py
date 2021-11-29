from enum import Enum
from typing import List, Any, Dict, Optional, Iterable

import logging
import numpy as np

from gps.agent import Agent
from gps.utility.labeled_data_packer import LabeledDataPacker
from gps.utility.timer import Timer
from vrep_interface.allegro import AllegroHand, AllegroFingerType
from vrep_interface import Framework, SimulationClient
from vrep_interface.simulation_object import RevoluteJoint


class Properties(Enum):
    JOINT_ANGLES = 0
    JOINT_VELOCITIES = 1
    TIP_POSITION = 2
    CYLINDER_ANGLE = 3
    CYLINDER_VELOCITY = 4
    TACTILE_FORCES = 5


class AgentAllegroVrep(Agent):
    """
    Implementation of the Agent interface for Allegro simulation using vrep.
    """

    def __init__(self, hand: AllegroHand, time_steps: int, control_cycle_time_ms: int, action_delay_ms: int,
                 used_fingers: List[AllegroFingerType], initial_states: List[Dict[AllegroFingerType, np.ndarray]],
                 simulation_client: SimulationClient, target_indicator_hand: Optional[AllegroHand] = None,
                 target_states: Optional[List[Dict[AllegroFingerType, np.ndarray]]] = None,
                 state_properties: Optional[Iterable[Properties]] = None,
                 observation_properties: Optional[Iterable[Properties]] = None):
        self._cylinder_joint = RevoluteJoint("cylinder_joint", online_writable=False, offline_writable=False)
        simulation_objects = list(hand.simulation_objects) + [self._cylinder_joint]
        self._use_target_indicator = target_indicator_hand is not None and target_states is not None
        if self._use_target_indicator:
            simulation_objects += [so for so in target_indicator_hand.simulation_objects if
                                   isinstance(so, RevoluteJoint)]
        self._framework = Framework(simulation_client, simulation_objects, control_cycle_time_ms,
                                    action_delay_ms)
        self._logger = logging.getLogger(__name__)
        self._finger_types = used_fingers
        self._hand = hand
        self._target_indicator_hand = target_indicator_hand
        self._target_states = target_states
        if (self._target_states is None) != (self._target_indicator_hand is None):
            self._logger.warning("Only one out of target_states and target_indicator_hand has been provided. The target"
                                 "indicator can thus not be used.")

        finger_labels = [((jp, t), 4) for t in self._finger_types for jp in
                         [Properties.JOINT_ANGLES, Properties.JOINT_VELOCITIES]] + \
                        [((Properties.TACTILE_FORCES, t), 23) for t in self._finger_types] + \
                        [((Properties.TIP_POSITION, t), 3) for t in self._finger_types]
        single_labels = [(Properties.CYLINDER_ANGLE, 1), (Properties.CYLINDER_VELOCITY, 1)]

        # By default, use all labels
        observation_properties = Properties if observation_properties is None else observation_properties
        state_properties = Properties if state_properties is None else state_properties

        state_labels = [l for l in single_labels if l[0] in state_properties] + \
                       [l for l in finger_labels if l[0][0] in state_properties]

        observation_labels = [l for l in single_labels if l[0] in observation_properties] + \
                             [l for l in finger_labels if l[0][0] in observation_properties]

        state_packer = LabeledDataPacker(state_labels)
        observation_packer = LabeledDataPacker(observation_labels)
        action_dimensions = 4 * len(used_fingers)
        super(AgentAllegroVrep, self).__init__(
            time_steps, len(initial_states), action_dimensions,
            state_packer=state_packer,
            observation_packer=observation_packer,
            tracking_point_labels=[(Properties.TIP_POSITION, t) for t in self._finger_types])

        self._initial_states = initial_states

        # List for collecting timing information
        # pre act delay; post act delay; pre observe delay; post observe delay
        self._timing_seconds = np.zeros((time_steps - 1, 3))
        self._timer = Timer()

    def _on_initialize(self):
        self._logger.info("Initializing framework...")
        self._framework.initialize()
        self._logger.info("Framework initialized.")

    def _reset(self, condition: int) -> Dict[Any, np.ndarray]:
        initial_state = self._initial_states[condition]
        for t in self._finger_types:
            self._hand.fingers[t].torques = 0, 0, 0, 0
            self._hand.fingers[t].initial_angles = initial_state[t]
            if self._use_target_indicator:
                self._target_indicator_hand.fingers[t].initial_angles = self._target_states[condition][t]
        self._framework.start()
        self._framework.observe()
        return self._get_state()

    def _do_step(self, action: np.ndarray, time_step: int) -> Dict[Any, np.ndarray]:
        for i, t in enumerate(self._finger_types):
            self._hand.fingers[t].torques = action[i * 4: (i + 1) * 4]
        self._timing_seconds[time_step, 0] = self._timer.round()
        self._framework.act()
        self._timing_seconds[time_step, 1] = self._timer.round()
        self._framework.observe()
        self._timing_seconds[time_step, 2] = self._timer.round()
        return self._get_state()

    def _on_sample_complete(self):
        # Evaluate timing
        self._framework.stop()
        policy_computation_delays = self._timing_seconds[1:, 0]
        act_delays = self._timing_seconds[:, 1]
        observe_delays = self._timing_seconds[:, 2]
        self._logger.debug(
            """Timing:
                    Policy comp.: avg: {:0.6f}s, min: {:0.6f}s ({:4d}), max: {:0.6f}s ({:4d})
                    Act         : avg: {:0.6f}s, min: {:0.6f}s ({:4d}), max: {:0.6f}s ({:4d})
                    Observe     : avg: {:0.6f}s, min: {:0.6f}s ({:4d}), max: {:0.6f}s ({:4d})
            """.format(*[f(t) for t in [policy_computation_delays, act_delays, observe_delays] for f in
                         [np.average, np.min, np.argmin, np.max, np.argmax]]))

    def _on_terminate(self):
        self._framework.terminate()

    def _get_state(self):
        state = {}
        f = self._hand.fingers
        state.update({(Properties.JOINT_ANGLES, t): f[t].angles for t in self._finger_types})
        state.update({(Properties.JOINT_VELOCITIES, t): f[t].angular_velocities for t in self._finger_types})
        state.update({(Properties.TIP_POSITION, t): f[t].tip_position for t in self._finger_types})
        state.update({(Properties.TACTILE_FORCES, t): f[t].tactile_forces for t in self._finger_types})
        state[Properties.CYLINDER_ANGLE] = np.array([self._cylinder_joint.observed_angle])
        state[Properties.CYLINDER_VELOCITY] = np.array([self._cylinder_joint.angular_velocity])
        return state
