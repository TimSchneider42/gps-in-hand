import os

import pickle
from typing import Dict, Any, Optional, List, Tuple

import logging

from setuptools import glob
from enum import Enum

from simplejson import OrderedDict

from allegro_pybullet.util import ReadOnlyOrderedDict
from gps.algorithm.iteration_data import ConditionData, Trajectory, TrajectoryCost, IterationData, AlgorithmData, \
    AlgorithmConditionData

LOGGER = logging.getLogger(__name__)


def _extract_reduced_trajectory(trajectory: Optional[Trajectory]) -> Optional[Trajectory]:
    """
    Returns a reduced version of the given trajectory.
    :param trajectory: Trajectory to reduce.
    :return: The reduced trajectory.
    """
    if trajectory is None:
        return None
    cost = [TrajectoryCost(tc.l, *([None] * 8)) for tc in trajectory.cost]
    return Trajectory(None, None, cost, trajectory.mean_cost, trajectory.samples, None, None, trajectory.expected_mu,
                      trajectory.expected_sigma, None, None, None)


def _reduce_iteration_data(iteration_data: IterationData) -> IterationData:
    """
    Returnes the reduced version of a IterationData object.
    :param iteration_data: The IterationData object to reduce
    :return: The reduced version of the given IterationData object
    """
    reduced_cond_data = {
        c: ConditionData(
            ReducedAlgorithmData(cd.algorithm_data.display_data) if cd.algorithm_data is not None else None,
            None,
            _extract_reduced_trajectory(cd.training_trajectory),
            _extract_reduced_trajectory(cd.test_trajectory),
            _extract_reduced_trajectory(cd.policy_trajectory))
        for c, cd in iteration_data.cond_data.items()}
    reduced_algorithm_data = ReducedAlgorithmData(iteration_data.algorithm_data.display_data)
    output = IterationData(iteration_data.iteration_no, reduced_algorithm_data, reduced_cond_data)
    return output


def _iteration_no_from_filename(filename):
    return int(os.path.split(filename)[-1].split(".")[0].split("_")[-1])


class ReducedAlgorithmData(AlgorithmData, AlgorithmConditionData):
    def __init__(self, display_data: Dict[str, Any]):
        self.__display_data = display_data
        AlgorithmConditionData.__init__(self, None)

    @property
    def display_data(self):
        return self.__display_data


class EntryState(Enum):
    NOT_LOADED = 0
    REDUCED_LOADED = 1
    LOADED = 2


class IterationDatabaseEntry:
    def __init__(self, file_path: str, iteration_data: Optional[IterationData] = None):
        self.__file_path = file_path
        self.__iteration_no = _iteration_no_from_filename(file_path)
        self.__iteration_data = iteration_data
        self.__state = EntryState.NOT_LOADED if iteration_data is None else EntryState.LOADED

    def __load_reduced(self) -> IterationData:
        # Load cached objects
        cache_file_name = self.__file_path + ".cache"
        if os.path.exists(cache_file_name):
            with open(cache_file_name, "rb") as f:
                return pickle.load(f)
        else:
            full_data = self.__load()
            self.__iteration_data = _reduce_iteration_data(full_data)

    def __load(self) -> IterationData:
        with open(self.__file_path, "rb") as f:
            return pickle.load(f)

    @property
    def iteration_data(self) -> IterationData:
        return self.__iteration_data

    @property
    def state(self) -> EntryState:
        return self.__state

    @state.setter
    def state(self, value):
        if value == EntryState.NOT_LOADED:
            if self.__state in [EntryState.REDUCED_LOADED, EntryState.LOADED]:
                self.__iteration_data = None
        elif value == EntryState.LOADED:
            if self.__state in [EntryState.NOT_LOADED, EntryState.REDUCED_LOADED]:
                self.__iteration_data = self.__load()
        elif value == EntryState.REDUCED_LOADED:
            if self.__state == EntryState.LOADED:
                self.__iteration_data = _reduce_iteration_data(self.__iteration_data)
            elif self.__state == EntryState.NOT_LOADED:
                self.__iteration_data = self.__load_reduced()
        self.__state = value

    @property
    def iteration_no(self) -> int:
        return self.__iteration_no

    @property
    def full_data_available(self) -> bool:
        return os.path.exists(self.__file_path)

    def __lt__(self, other: "IterationDatabaseEntry"):
        return self.iteration_no < other.iteration_no


class IterationDatabase:
    """
    Stores and loads IterationData from the memory. In order to make that memory and runtime efficient, the database
    will keep reduced versions of the data in the RAM. The following attributes are present in reduced IterationData
    versions:
    - iteration number
    - cost over time for each sample
    - expected trajectories for each condition
    - tracking point trajectories for each sample

    """

    def __init__(self, directory: str, keep_newest_only: bool = True, create_caches: bool = True,
                 default_entry_state: EntryState = EntryState.REDUCED_LOADED):
        """

        :param directory:           Storage directory of the database.
        :param keep_newest_only:    True, if only the newest IterationData object is to be kept. As soon as a new object
                                    is stored, all old objects will be deleted. Caches however - if created - will be
                                    kept.
        :param create_caches:       True, if the database shall create caches. Caches speed up the loading of
                                    IterationData objects significantly.
        :param default_entry_state: Default state of the database entries.
        """
        self.__directory = os.path.abspath(directory)
        self.__keep_newest_only = keep_newest_only
        self.__create_caches = create_caches
        self.__full_iteration_numbers: Optional[List[int]] = None
        self.__entries: Optional[OrderedDict[int, IterationDatabaseEntry]] = None
        self.__default_entry_state = default_entry_state

    def initialize_new(self):
        """
        Initializes an empty database.
        :return:
        """
        os.makedirs(self.__directory)
        self.__full_iteration_numbers = []
        self.__entries = OrderedDict()

    def load(self):
        """
        Loads an existing database.
        :return:
        """
        try:
            # Load full objects
            itr_data_files = glob.glob(os.path.join(self.__directory, "iteration_[0-9][0-9][0-9].pkl"))
            self.__full_iteration_numbers = sorted([_iteration_no_from_filename(f) for f in itr_data_files])

            # Load cached objects
            itr_data_cache_files = glob.glob(os.path.join(self.__directory, "iteration_[0-9][0-9][0-9].pkl.cache"))
            iteration_numbers = set(
                self.__full_iteration_numbers + [_iteration_no_from_filename(f) for f in itr_data_cache_files])

            self.__entries = OrderedDict(
                [(d.iteration_no, d) for d in
                 sorted([IterationDatabaseEntry(self.__filename_of(i)) for i in iteration_numbers])])
            for e in self.__entries.values():
                e.state = self.__default_entry_state
        except Exception:
            self.__full_iteration_numbers = None
            self.__entries = None
            raise

    def store(self, iteration_data: IterationData):
        """
        Stores a new IterationData object in the database
        :param iteration_data:  The new IterationData object
        :param entry_state:     Target state of the stored entry.
        :return:
        """
        assert self.initialized, "This database has not been initialized."
        file_path = self.__filename_of(iteration_data.iteration_no)
        if self.__create_caches:
            reduced_data = _reduce_iteration_data(iteration_data)
            with open(file_path + ".cache", "wb") as f:
                pickle.dump(reduced_data, f)
        if not self.__keep_newest_only or len(self.__full_iteration_numbers) == 0 or iteration_data.iteration_no > \
                self.__full_iteration_numbers[-1]:
            with open(file_path, "wb") as f:
                pickle.dump(iteration_data, f)
            self.__full_iteration_numbers.append(iteration_data.iteration_no)
        if self.__keep_newest_only:
            for n in self.__full_iteration_numbers[:-1]:
                os.remove(self.__filename_of(n))
            self.__full_iteration_numbers = [self.__full_iteration_numbers[-1]]
        entry = IterationDatabaseEntry(file_path, iteration_data)
        entry.state = self.__default_entry_state
        self.__entries[entry.iteration_no] = entry
        return entry

    @property
    def full_iteration_numbers(self) -> Optional[List[int]]:
        """
        Numbers of all currently present full IterationData objects
        :return:
        """
        return self.__full_iteration_numbers

    @property
    def entries(self) -> Optional[ReadOnlyOrderedDict[int, IterationDatabaseEntry]]:
        """
        All currently present database entries.
        :return:
        """
        return ReadOnlyOrderedDict(self.__entries)

    @property
    def initialized(self) -> bool:
        """
        True, if the database has been initialized
        :return:
        """
        return self.__entries is not None

    def __filename_of(self, iteration_number: int):
        """
        Returns the filename of the IterationData object with the given number in the database.
        :param iteration_number: The number of the IterationData object
        :return: The full path to the requested object file.
        """
        return os.path.join(self.__directory, f"iteration_{iteration_number:03d}.pkl")
