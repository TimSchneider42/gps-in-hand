from typing import Iterable

import numpy as np

from gps.sample import Sample


class SampleList:
    """
    Provides a simplified access to data of multiple samples.
    """

    def __init__(self, samples: Iterable[Sample]):
        """

        :param samples: List of samples to use.
        """
        self._samples = tuple(samples)
        self._states = np.array([s.states for s in self])
        self._actions = np.array([s.actions for s in self])
        self._observations = np.array([s.observations for s in self])

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return self._samples.__iter__()

    def __getitem__(self, item: int) -> Sample:
        return self._samples.__getitem__(item)

    @property
    def states(self):
        """
        N x T x dX numpy array containing all sample's states.
        :return:
        """
        return self._states

    @property
    def actions(self):
        """
        N x T x dX numpy array containing all sample's actions.
        :return:
        """
        return self._actions

    @property
    def observations(self):
        """
        N x T x dX numpy array containing all sample's observations.
        :return:
        """
        return self._observations
