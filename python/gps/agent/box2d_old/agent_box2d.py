""" This file defines an agent for the Box2D simulator. """
from typing import List, Any, Dict

import numpy as np
from gps.agent.agent import Agent
from gps.agent.box2d.framework import Framework
from gps.utility.labeled_data_packer import LabeledDataPacker


class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """

    def __init__(self, time_steps: int, initial_states: List[Dict[Any, np.ndarray]], framework: Framework,
                 state_data_types: List[Any], observation_data_types: List[Any], tracking_data_types: List[Any]):
        state_packer = LabeledDataPacker([("angles", 2), ("vel", 2), ("tracking", 3)])
        super(AgentBox2D, self).__init__(time_steps, len(initial_states), 2, state_packer,
                                         state_packer, tracking_data_types)

        self._framework = framework
        self._initial_states = initial_states

    def _reset(self, condition: int) -> Dict[Any, np.ndarray]:
        self._framework.reset_world()
        self._framework.run()
        return self._framework.get_state()

    def _do_step(self, action: np.array, time_step: int) -> Dict[Any, np.ndarray]:
        self._framework.run_next([np.asscalar(a) for a in action])
        return self._framework.get_state()

    @property
    def tracking_point_labels(self):
        return ["tracking"]