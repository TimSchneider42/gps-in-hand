"""
GPS Training GUI

The GPS Training GUI is used to interact with the GPS algorithm during training.
It contains the below seven functionalities:

Action Panel                contains buttons for stop, reset, go, fail
Action Status Textbox       displays action status
Algorithm Status Textbox    displays algorithm status
Cost Plot                   displays costs after each iteration
Algorithm Output Textbox    displays algorithm output after each iteration
3D Trajectory Visualizer    displays 3D trajectories after each iteration
Image Visualizer            displays images received from a rostopic

For more detailed documentation, visit: rll.berkeley.edu/gps/gui
"""
import os
import signal
import time
from queue import Queue, Empty
from typing import List, Tuple, Optional, Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from threading import Thread

from gps import Experiment
from gps.algorithm import IterationData
from gps.gui.textbox import Textbox
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.multi_trajectory_plotter import MultiTrajectoryPlotter


class OnlineGUI:
    def __init__(self, experiment: Experiment, run_name: str, iteration: Optional[int] = None,
                 quit_on_end: bool = False, wait_on_start: bool = True, figsize: Tuple[int, int] = (12, 12)):
        self._experiment = experiment
        self._quit_on_end = quit_on_end
        self._wait_on_start = wait_on_start

        self._status_colors = {
            Experiment.Status.NOT_STARTED: "grey",
            Experiment.Status.TRAINING_SAMPLING: "cyan",
            Experiment.Status.CALCULATING: "cyan",
            Experiment.Status.TEST_SAMPLING: "cyan",
            Experiment.Status.POLICY_SAMPLING: "cyan",
            Experiment.Status.ITERATION_DONE: "cyan",
            Experiment.Status.WAITING: "orange",
            Experiment.Status.ABORTED: "firebrick",
            Experiment.Status.DONE: "green",
            Experiment.Status.CRASHED: "red"
        }

        self._status_display_names = {
            Experiment.Status.NOT_STARTED: "Not started",
            Experiment.Status.TRAINING_SAMPLING: "Training sampling...",
            Experiment.Status.CALCULATING: "Calculating...",
            Experiment.Status.TEST_SAMPLING: "Test sampling...",
            Experiment.Status.POLICY_SAMPLING: "Policy sampling...",
            Experiment.Status.ITERATION_DONE: "Iteration done",
            Experiment.Status.WAITING: "Waiting...",
            Experiment.Status.ABORTED: "Aborted",
            Experiment.Status.DONE: "Done",
            Experiment.Status.CRASHED: "Crashed"
        }

        # Setup figure.
        plt.rcParams["toolbar"] = "None"
        for key in plt.rcParams:
            if key.startswith("keymap."):
                plt.rcParams[key] = ""

        self._fig = plt.figure(figsize=figsize)
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                  wspace=0, hspace=0)
        self._fig.canvas.mpl_connect("close_event", self._on_figure_closed)
        self._fig.canvas.set_window_title("Experiment {}, Run {}".format(experiment.name, run_name))

        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(16, 8)
        self._gs_action_output = self._gs[0:1, 0:3]
        self._gs_status_output = self._gs[1:2, 0:3]
        self._gs_cost_plotter = self._gs[0:2, 4:8]
        self._gs_algthm_output = self._gs[2:8, 0:8]
        self._gs_traj_visualizer = self._gs[8:16, 0:8]

        # Create GUI components.
        self._status_output = Textbox(self._fig, self._gs_action_output, border_on=True)
        self._experiment_status_output = Textbox(self._fig, self._gs_status_output, border_on=False)
        self._algthm_output = Textbox(self._fig, self._gs_algthm_output,
                                      max_display_size=15,
                                      fontsize=10,
                                      font_family="monospace")

        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_cost_plotter)
        ax = plt.subplot(gs[0])

        algo = self._experiment.algorithm
        self._include_test_data = len(algo.test_conditions) > 0 and algo.test_samples_per_condition > 0
        self._include_policy_data = len(algo.policy_conditions) > 0 and algo.policy_samples_per_condition > 0
        labels = ["training"]
        conditions = set(algo.training_conditions)
        if self._include_test_data:
            labels.append("test")
            conditions.update(algo.test_conditions)

        if self._include_policy_data:
            labels.append("policy")
            conditions.update(algo.policy_conditions)

        self._cost_plotter = MeanPlotter(ax, labels=labels)

        self._traj_visualizer = MultiTrajectoryPlotter(self._fig, self._gs_traj_visualizer, conditions=conditions,
                                                       tracking_point_labels=self._experiment.agent.tracking_point_labels,
                                                       state_packer=self._experiment.agent.state_packer,
                                                       width_height_ratio=2.0)

        self._quit = False
        experiment.setup(run_name=run_name, itr_load=iteration)
        self._iteration_data_queue = Queue()
        self._status_queue = Queue()
        self._status_queue.put((experiment.status, experiment.status_message))
        # Used to invoke function calls on the experiment object
        # format: (function, args, kwargs)
        self._invocation_queue = Queue()
        # Queue to return data from invocations
        self._return_queue = Queue()
        self._experiment_thread = Thread(target=OnlineGUI._run_experiment,
                                         args=(experiment, self._invocation_queue, self._return_queue,
                                             self._status_queue, self._iteration_data_queue))
        self._condition_data_description: List[Tuple[str, str, Callable[[float], str], int]] = None
        self._data_description: List[Tuple[str, str, Callable[[Any], str], int]] = None
        self._condition_data_length = {}
        self._update_interval = 0.1
        self._output_text_header = ""
        self._output_text_lines = []

        # Variables for mean plotter
        self._cost_data = None
        self._iterations = None

    @staticmethod
    def _run_experiment(experiment: Experiment, invocation_queue: Queue, return_queue: Queue, status_queue,
                        iteration_data_queue):
        experiment.status_changed_event.add(OnlineGUI._put_in_queue_func(status_queue))
        experiment.iteration_done_event.add(OnlineGUI._put_in_queue_func(iteration_data_queue))
        experiment_thread = Thread(target=experiment.run)
        experiment_thread.start()
        while experiment_thread.is_alive():
            try:
                func, args, kwargs = invocation_queue.get(timeout=0.2)
                return_queue.put(func(experiment, *args, **kwargs))
            except Empty:
                pass

    def _invoke_on_experiment(self, func: Callable, *args, **kwargs):
        self._invocation_queue.put((func, args, kwargs))
        return self._return_queue.get()

    @staticmethod
    def _put_in_queue_func(queue: Queue):
        def f(sender, *data):
            if len(data) == 1:
                queue.put(data[0])
            else:
                queue.put(data)

        return f

    def show(self):
        """
        Displays the GUI. Please note that this function is blocking.
        :return:
        """
        try:
            for s in [signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGABRT]:
                signal.signal(s, self._signal_handler)
            self._quit = False
            self._initialize()
            plt.ioff()
            self._fig.show()
            next_update = time.time()
            while not self._quit:
                # Run main update loop
                sleep_time = max(next_update - time.time(), 0.0)
                time.sleep(sleep_time)
                self._update()
                next_update += self._update_interval
        finally:
            self._terminate()

    def quit(self):
        self._quit = True

    def _signal_handler(self, signum, frame):
        self.quit()

    def _initialize(self):
        # Setup GUI components.
        self._set_output_text(self._experiment.info)
        self._prepare_description()
        self._cost_data = [[] for i in [self._include_policy_data, self._include_test_data] if i] + [[]]
        self._iterations = []
        self._old_status = None
        if not self._wait_on_start:
            self._start_experiment_proc()

    def _update(self):
        status = self._refresh_status()
        self._handle_iteration_summaries()
        self._fig.canvas.update()
        self._fig.canvas.flush_events()  # Fixes bug with Qt4Agg backend
        if status in [Experiment.Status.DONE, Experiment.Status.CRASHED,
                      Experiment.Status.ABORTED] and self._quit_on_end:
            self.quit()

    def _start_experiment_proc(self):
        self._experiment_thread.start()

    def _terminate(self):
        if self._experiment_thread.is_alive():
            self._invoke_on_experiment(Experiment.abort)
            self._experiment_thread.join()

    def _action_go(self, event):
        if not self._experiment_thread.is_alive():
            self._start_experiment_proc()
        else:
            self._invoke_on_experiment(Experiment.resume)

    def _action_stop(self, event):
        pass

    def _on_figure_closed(self, event):
        self.quit()

    def _refresh_status(self):
        try:
            status, message = self._status_queue.get_nowait()
            self._status_output.set_text(self._status_display_names[status])
            self._status_output.set_bgcolor(self._status_colors[status], 1.0)
            if message is not None:
                self._experiment_status_output.set_text(message)
            else:
                self._experiment_status_output.set_text("")
            self._fig.canvas.draw()
            return status
        except Empty:
            return None

    def _set_output_text(self, text):
        self._algthm_output.set_text(text)

    def _append_output_text(self, text):
        self._algthm_output.append_text(text)

    def _handle_iteration_summaries(self):
        while not self._iteration_data_queue.empty():
            training_conditions = self._experiment.algorithm.training_conditions
            test_conditions = self._experiment.algorithm.test_conditions
            policy_conditions = self._experiment.algorithm.policy_conditions

            itr_data: IterationData = self._iteration_data_queue.get_nowait()
            self._iterations.append(itr_data.iteration_no)

            training_cost = np.array([itr_data.cond_data[c].training_trajectory.mean_cost for c in training_conditions])
            self._cost_data[0].append(training_cost)

            if self._include_test_data:
                test_cost = np.array([itr_data.cond_data[c].test_trajectory.mean_cost for c in test_conditions])
                self._cost_data[1].append(test_cost)

            if self._include_policy_data:
                policy_cost = np.array([itr_data.cond_data[c].policy_trajectory.mean_cost for c in policy_conditions])
                self._cost_data[2].append(policy_cost)
            self._cost_plotter.update(self._iterations, self._cost_data)

            self._update_iteration_data(itr_data)

            test_trajectories = {c: itr_data.cond_data[c].test_trajectory for c in test_conditions}
            training_trajectories = {c: itr_data.cond_data[c].training_trajectory for c in training_conditions}
            policy_trajectories = {c: itr_data.cond_data[c].policy_trajectory for c in policy_conditions}
            self._traj_visualizer.update_trajectories(training_trajectories, test_trajectories, policy_trajectories)
            self._fig.canvas.draw()

    def _prepare_description(self):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        self._set_output_text(self._experiment.name)
        algorithm = self._experiment.algorithm

        self._data_description = [d[1:] for d in algorithm.display_data_description if not d[0]]
        self._condition_data_description = [d[1:] for d in algorithm.display_data_description if d[0]]

        itr_data_fields = " ".join([f"{n: >{c}}" for i, n, f, c in self._data_description])
        condition_titles = " " * len(itr_data_fields)

        cond_fields = " ".join([f"{n: >{c}}" for i, n, f, c in self._condition_data_description])

        for cond in sorted(set(algorithm.training_conditions).union(algorithm.test_conditions)):
            cond_title = "condition {}".format(cond)

            length = max(len(cond_fields), len(cond_title))
            self._condition_data_length[cond] = length

            condition_titles += " | " + cond_title.ljust(length)
            itr_data_fields += " | " + cond_fields.ljust(length)
        self._output_text_header = condition_titles + os.linesep + itr_data_fields
        self._output_text_lines = []
        self._set_output_text(self._output_text_header)

    def _update_iteration_data(self, itr_data: IterationData):
        """
        Update iteration data information.
        :param itr_data: Iteration data of current iteration.
        :return:
        """
        disp = itr_data.display_data
        data = " ".join([f(disp[i]).ljust(c) for i, n, f, c in self._data_description])
        algorithm = self._experiment.algorithm
        for cond in sorted(set(algorithm.training_conditions).union(algorithm.test_conditions)):
            cond_disp = itr_data.cond_data[cond].display_data
            cond_data = " ".join([f(cond_disp[i]).ljust(c) for i, n, f, c in self._condition_data_description])
            data += " | " + cond_data.ljust(self._condition_data_length[cond])
        self._output_text_lines.append(data)
        self._set_output_text(os.linesep.join([self._output_text_header] + self._output_text_lines[-10:]))
