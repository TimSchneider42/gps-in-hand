import signal
import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.experiment import Experiment
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.multi_trajectory_plotter import MultiTrajectoryPlotter, DetailLevel
from gps.utility.iteration_database import IterationDatabase


class OfflineGUI:
    def __init__(self, iteration_database: IterationDatabase, experiment: Experiment, run_name: str,
                 figsize: Tuple[int, int] = (12, 12), detail_level: DetailLevel = DetailLevel.NO_DETAILS,
                 conditions: List[int] = None):
        assert len(iteration_database.reduced_iteration_data) > 0
        self._experiment = experiment
        self._run_name = run_name
        self._iteration_database = iteration_database

        itr_data0 = iteration_database.reduced_iteration_data[0]

        self._training_conditions = [c for c, cd in itr_data0.cond_data.items() if
                                     cd.training_trajectory is not None and (conditions is None or c in conditions)]
        self._test_conditions = [c for c, cd in itr_data0.cond_data.items() if
                                 cd.test_trajectory is not None and (conditions is None or c in conditions)]
        self._policy_conditions = [c for c, cd in itr_data0.cond_data.items() if
                                   cd.policy_trajectory is not None and (conditions is None or c in conditions)]

        self._fig = plt.figure(figsize=figsize)
        self._fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.93)
        self._fig.canvas.mpl_connect("close_event", self._on_figure_closed)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_pressed)

        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(16, 1)
        self._gs_cost_plotter = self._gs[:2, 0]
        self._gs_traj_visualizer = self._gs[3:, 0]

        # Create GUI components.
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_cost_plotter)
        ax = plt.subplot(gs[0])
        self._include_test_data = len(self._test_conditions) > 0
        self._include_policy_data = len(self._policy_conditions) > 0

        labels = ["training"]
        conditions = set(self._training_conditions)
        if self._include_test_data:
            labels.append("test")
            conditions.update(self._test_conditions)
        if self._include_policy_data:
            labels.append("policy")
            conditions.update(self._policy_conditions)
        self._cost_plotter = MeanPlotter(ax, labels=labels)

        self._traj_visualizer = MultiTrajectoryPlotter(self._fig, self._gs_traj_visualizer, conditions=conditions,
                                                       tracking_point_labels=experiment.agent.tracking_point_labels,
                                                       state_packer=experiment.agent.state_packer,
                                                       width_height_ratio=2.0, detail_level=detail_level)

        self._quit = False

        self._update_interval = 0.1

        self._iteration_index = None
        self._current_iteration = None

        # Load cost data
        cost = [np.array([
            [itr.cond_data[c].training_trajectory.mean_cost for c in self._training_conditions] for itr
            in iteration_database.reduced_iteration_data])]
        if self._include_test_data:
            cost.append(np.array([
                [itr.cond_data[c].test_trajectory.mean_cost for c in self._test_conditions] for itr in
                iteration_database.reduced_iteration_data]))
        if self._include_policy_data:
            cost.append(np.array([
                [itr.cond_data[c].policy_trajectory.mean_cost for c in self._policy_conditions] for itr in
                iteration_database.reduced_iteration_data]))
        iterations = np.array([itr.iteration_no for itr in iteration_database.reduced_iteration_data])
        self._cost_plotter.update(iterations, cost)

        self._set_iteration(0)

        # Connect mouse click event
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Index of iteration number in list
        self._index_of = {it.iteration_no: i for i, it in enumerate(iteration_database.reduced_iteration_data)}

    def _on_click(self, event):
        if event.inaxes == self._cost_plotter.axes:
            iteration = int(event.xdata + 0.5)
            self._set_iteration(
                self._index_of[max(0, min(len(self._iteration_database.reduced_iteration_data), iteration))])

    def _set_iteration(self, iteration_index: int):
        if iteration_index != self._iteration_index:
            if self._current_iteration is not None and \
                    self._traj_visualizer.detail_level == DetailLevel.COST_AND_POLICY:
                del self._current_iteration
            self._iteration_index = iteration_index
            iteration = self._iteration_database.reduced_iteration_data[iteration_index]
            if self._traj_visualizer.detail_level == DetailLevel.COST_AND_POLICY:
                iteration = self._iteration_database.load_iteration_full(iteration.iteration_no)
            self._fig.suptitle(
                "Experiment \"{}\", run \"{}\", iteration {}".format(self._experiment.name, self._run_name,
                                                                     iteration.iteration_no))
            self._cost_plotter.highlight(iteration.iteration_no)

            test_trajectories = {c: iteration.cond_data[c].test_trajectory for c in self._test_conditions}
            training_trajectories = {c: iteration.cond_data[c].training_trajectory for c in self._training_conditions}
            policy_trajectories = {c: iteration.cond_data[c].policy_trajectory for c in self._policy_conditions}
            self._current_iteration = iteration
            self._traj_visualizer.update_trajectories(training_trajectories, test_trajectories, policy_trajectories)
            self._fig.canvas.draw()

    def show(self):
        """
        Displays the GUI. Please note that this function is blocking.
        :return:
        """
        for s in [signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGABRT]:
            signal.signal(s, self._signal_handler)
        self._quit = False
        plt.ioff()
        self._fig.show()
        next_update = time.time()
        while not self._quit:
            # Run main update loop
            sleep_time = max(next_update - time.time(), 0.0)
            time.sleep(sleep_time)
            self._update()
            next_update += self._update_interval

    def quit(self):
        self._quit = True

    def _signal_handler(self, signum, frame):
        self.quit()

    def _update(self):
        self._fig.canvas.update()
        self._fig.canvas.flush_events()  # Fixes bug with Qt4Agg backend

    def _on_figure_closed(self, event):
        self.quit()

    def _on_key_pressed(self, event):
        if event.key == "right":
            self._set_iteration((self._iteration_index + 1) % len(self._iteration_database.reduced_iteration_data))
        elif event.key == "left":
            self._set_iteration((self._iteration_index - 1) % len(self._iteration_database.reduced_iteration_data))
