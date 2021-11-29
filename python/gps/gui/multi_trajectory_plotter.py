from typing import Dict, List, Any, Iterable

import math

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from gps.algorithm import Trajectory
from gps.gui.trajectory_plotter import TrajectoryPlotter, DetailLevel
from gps.utility.labeled_data_packer import LabeledDataPacker


class MultiTrajectoryPlotter:
    def __init__(self, figure: Figure, subplot_spec: SubplotSpec, conditions: Iterable[int],
                 tracking_point_labels: List[Any], state_packer: LabeledDataPacker, width_height_ratio=1.0,
                 detail_level: DetailLevel = DetailLevel.NO_DETAILS):
        self._state_packer = state_packer
        self._tracking_point_labels = tracking_point_labels

        condition = list(conditions)

        rows = int(round(math.sqrt(len(condition) / width_height_ratio)))
        cols = int(math.ceil(len(condition) / rows))

        self._gs_plots = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=subplot_spec)
        self._trajectory_plotters: Dict[int, TrajectoryPlotter] = {
            c: TrajectoryPlotter(figure, self._gs_plots[i // cols, i % cols], tracking_point_labels, state_packer,
                                 title="Condition {}".format(c), show_legend=True, detail_level=detail_level) for i, c
            in enumerate(condition)}

        self._detail_level = detail_level

    def update_trajectories(self, training_trajectories: Dict[int, Trajectory],
                            test_trajectories: Dict[int, Trajectory], policy_trajectories: Dict[int, Trajectory]):
        lims = np.array([[np.inf] * 3, [-np.inf] * 3])
        cost_lims = np.array([np.inf, -np.inf])
        for cond, traj_plotter in self._trajectory_plotters.items():
            # Don't refresh now as the limits are not yet set
            traj_plotter.update_trajectories(
                training_trajectories.get(cond), test_trajectories.get(cond), policy_trajectories.get(cond),
                refresh_plots=False)

            # Compute limits to be equal for each plot
            local_lims = traj_plotter.lims3d
            lims[0] = np.minimum(lims[0], local_lims.T[0])
            lims[1] = np.maximum(lims[1], local_lims.T[1])

            local_cost_lims = traj_plotter.lims_cost
            cost_lims[0] = min(cost_lims[0], local_cost_lims[0])
            cost_lims[1] = max(cost_lims[1], local_cost_lims[1])

        for traj_plotter in self._trajectory_plotters.values():
            traj_plotter.lims3d = lims.T
            traj_plotter.lims_cost = cost_lims
            traj_plotter.refresh_plots()

    @property
    def detail_level(self) -> DetailLevel:
        return self._detail_level
