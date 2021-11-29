from enum import Enum
from typing import List, Any, Dict

import logging
import numpy as np

from gps.agent.agent import Agent
from gps.utility.labeled_data_packer import LabeledDataPacker
from gps.utility.timer import Timer
from vrep_interface import Framework, SimulationClient
from vrep_interface.darias import DariasFinger, DariasHand


class Properties(Enum):
    JOINT_ANGLES = 0
    JOINT_VELOCITIES = 1
    TIP_POSITION = 2


class AgentDariasVrep(Agent):
    """
    Implementation of the Agent interface for Darias.
    """

    def __init__(self, hand: DariasHand, time_steps: int, control_cycle_time_ms: int, action_delay_ms: int,
                 used_fingers: List[DariasFinger], initial_states: List[Dict[DariasFinger, np.ndarray]],
                 simulation_client: SimulationClient):
        self._framework = Framework(simulation_client, hand.simulation_objects, control_cycle_time_ms,
                                    action_delay_ms)
        self._fingers = used_fingers

        state_packer = LabeledDataPacker(
            [((jp, f.name_prefix), 3) for f in used_fingers for jp in Properties] + [])
        action_dimensions = 3 * len(used_fingers)
        super(AgentDariasVrep, self).__init__(time_steps, len(initial_states), action_dimensions, state_packer,
                                              tracking_point_labels=[(Properties.TIP_POSITION, f.name_prefix) for f in
                                                                 used_fingers])

        self._initial_states = initial_states
        self._logger = logging.getLogger(__name__)

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
        for f in self._fingers:
            f.torques = 0, 0, 0
            f.initial_angles = initial_state[f]
        self._framework.start()
        self._framework.observe()
        return self._get_state()

    def _do_step(self, action: np.ndarray, time_step: int) -> Dict[Any, np.ndarray]:
        for i, f in enumerate(self._fingers):
            f.torques = action[i * 3: (i + 1) * 3]
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
        state.update({(Properties.JOINT_ANGLES, f.name_prefix): f.angles for f in self._fingers})
        state.update({(Properties.JOINT_VELOCITIES, f.name_prefix): f.angular_velocities for f in self._fingers})
        state.update({(Properties.TIP_POSITION, f.name_prefix): f.tip_position for f in self._fingers})
        return state
