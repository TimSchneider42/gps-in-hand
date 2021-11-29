""" This file defines the main object that runs experiments. """
import logging
from enum import Enum
from threading import Lock
from typing import Optional, List

from gps.agent import Agent, Sampler
from gps.algorithm import Algorithm
from gps.utility.abortable_worker import AbortableWorker
from gps.utility.iteration_database import IterationDatabase, EntryState
from gps.utility.event import Event
from gps.utility.status_message_handler import StatusMessageHandler


class Experiment(AbortableWorker[None]):
    """ Main class to run algorithms and experiments. """

    class Status(Enum):
        NOT_STARTED = 0
        WAITING = 1
        TRAINING_SAMPLING = 2
        CALCULATING = 3
        TEST_SAMPLING = 4
        POLICY_SAMPLING = 5
        DONE = 6
        ITERATION_DONE = 7
        ABORTED = 8
        CRASHED = 9

    def __init__(self, name: str, algorithm: Algorithm, num_iterations: int,
                 iteration_database: IterationDatabase, save_progress: bool = True):
        """

        :param name:                    Name of the experiment to run.
        :param algorithm:               Algorithm to use.
        :param num_iterations:          Total number of iterations.
        :param iteration_database:      Database to store IterationData objects in.
        :param save_progress:           True, if the algorithm's progress shall be saved.
        """
        AbortableWorker.__init__(self)
        self._status_changed_event = Event()
        self._iteration_done_event = Event()

        self._name = name
        self._iteration_database = iteration_database

        self._algorithm: Algorithm = algorithm

        self._num_iterations = num_iterations

        self._save_progress = save_progress

        self._status_message_handler = StatusMessageHandler("")

        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(self._status_message_handler)

        self._info = f"""
                    exp_name:   {self.name}
                    alg_type:   {self._algorithm.display_name}
                    iterations: {self._num_iterations}
                    conditions: {self.agent.condition_count}
                """

        self._abort_lock = Lock()

        self._current_test_samplers: List[Sampler] = []

        self._change_status(Experiment.Status.NOT_STARTED, "Experiment has not yet been started.")

    def _run(self, run_name: str, itr_load: int = None):
        """
        Run training by iteratively sampling and taking an iteration.
        :param run_name:    Name of this run. Run names are unique within each experiment. If this run does not exist,
                            it will be created. Otherwise it can be continued.
        :param itr_load:    If specified, loads algorithm state from that
                            iteration, and resumes training at the next iteration.
        :return:
        """
        try:
            self._algorithm.agent.initialize()
            itr = None

            def training_sample_started_handler(sender: Sampler, sample_no: int):
                total = self.algorithm.training_samples_per_condition
                self._change_status(Experiment.Status.TRAINING_SAMPLING,
                                    "Sampling (training): iteration {i}, condition {c}, sample {s}/{t}.".format(
                                        i=itr, c=sender.condition, s=sample_no + 1, t=total))

            def test_sample_started_handler(sender: Sampler, sample_no: int):
                total = self.algorithm.test_samples_per_condition
                self._change_status(Experiment.Status.TEST_SAMPLING,
                                    "Sampling (testing): iteration {i}, condition {c}, sample {s}/{t}.".format(
                                        i=itr, c=sender.condition, s=sample_no + 1, t=total))

            def policy_sample_started_handler(sender: Sampler, sample_no: int):
                total = self.algorithm.policy_samples_per_condition
                self._change_status(Experiment.Status.POLICY_SAMPLING,
                                    "Sampling (policy): iteration {i}, condition {c}, sample {s}/{t}.".format(
                                        i=itr, c=sender.condition, s=sample_no + 1, t=total))

            def sampling_completed_handler(sender: Sampler):
                self._change_status(Experiment.Status.CALCULATING, "Calculating...")

            for s in self._algorithm.test_samplers.values():
                s.sample_started_event.add(test_sample_started_handler)

            for s in self._algorithm.training_samplers.values():
                s.sample_started_event.add(training_sample_started_handler)
                s.sampling_completed_event.add(sampling_completed_handler)

            for s in self._algorithm.policy_samplers.values():
                s.sample_completed_event.add(policy_sample_started_handler)

            if itr_load is None:
                self._algorithm.initialize_new()
                iterations = range(self._num_iterations)
                self._logger.info("Initialized new training.")
            else:
                iterations = range(itr_load + 1, self._num_iterations)
                itr_data = self._iteration_database.entries[itr_load]
                itr_data.state = EntryState.LOADED
                self.algorithm.set_iteration(itr_data.iteration_data)

                self._logger.info("Resuming training from algorithm state at iteration {0}.".format(itr_load + 1))

            for itr in iterations:
                # Take algorithm iteration
                with self._abort_lock:
                    if self.aborting:
                        return
                    self._algorithm.setup()
                self._algorithm.run()

                if self.aborting:
                    return

                itr_data = self.algorithm.current_iteration

                # Save state
                self._iteration_database.store(itr_data).state = EntryState.NOT_LOADED
                self._change_status(Experiment.Status.ITERATION_DONE)
                self._iteration_done_event(self, self.algorithm.current_iteration)
            self._change_status(Experiment.Status.DONE, "Training complete.")
        except Exception:
            self._change_status(Experiment.Status.CRASHED, "Training crashed.")
            raise
        finally:
            self._algorithm.agent.terminate()

    def _change_status(self, new_status: "Experiment.Status", message: Optional[str] = None):
        self._status = new_status
        if message is not None:
            self._logger.info("Status {}: {}".format(new_status.name, message))
        else:
            self._logger.info("Status {}".format(new_status.name))
        self._status_changed_event(self, new_status, message)

    def _on_abort(self):
        """
        Abort the experiment and exit cleanly.
        :return:
        """
        with self._abort_lock:
            self._algorithm.abort()

    def stop(self):
        """
        Abort the current sample and pause the experiment without exiting.
        :return:
        """
        raise NotImplementedError()

    def resume(self):
        """
        Resume the experiment after it has been paused.
        :return:
        """
        raise NotImplementedError()

    @property
    def agent(self) -> Agent:
        return self._algorithm.agent

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    def info(self) -> str:
        return self._info

    @property
    def status(self) -> "Experiment.Status":
        return self._status

    @property
    def status_message(self) -> str:
        return self._status_message_handler.message

    @property
    def status_changed_event(self) -> Event:
        return self._status_changed_event

    @property
    def iteration_done_event(self) -> Event:
        return self._iteration_done_event

    @property
    def name(self) -> str:
        return self._name
