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
        assert all(s in framework.state_dimensions.keys() for s in state_data_types)
        assert all(o in framework.state_dimensions.keys() for o in observation_data_types)
        state_packer = LabeledDataPacker([(l, framework.state_dimensions[l]) for l in state_data_types])
        observation_packer = LabeledDataPacker([(l, framework.state_dimensions[l]) for l in observation_data_types])
        super(AgentBox2D, self).__init__(time_steps, len(initial_states), framework.action_dimensions, state_packer,
                                         observation_packer, tracking_data_types)

        self._framework = framework
        self._initial_states = initial_states

    def _reset(self, condition: int) -> Dict[Any, np.ndarray]:
        self._framework.reset_world(self._initial_states[condition])
        self._framework.run()
        return self._framework.get_state()

    def _do_step(self, action: np.array, time_step: int) -> Dict[Any, np.ndarray]:
        self._framework.run_next([np.asscalar(a) for a in action])
        return self._framework.get_state()