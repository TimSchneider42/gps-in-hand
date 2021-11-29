from enum import Enum
from typing import List, Any, Optional, Union, Tuple, Dict

import itertools
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.texmanager import TexManager
from mpl_toolkits.mplot3d import Axes3D

from gps.algorithm import Trajectory
from gps.controller import LinearGaussianController
from gps.gui.util import buffered_axis_limits
from gps.utility.labeled_data_packer import LabeledDataPacker


class DetailLevel(Enum):
    NO_DETAILS = 0
    COST = 1
    COST_AND_POLICY = 2


class TrajectoryPlotter:
    def __init__(self, figure: Figure, subplot_spec: SubplotSpec, tracking_point_labels: List[Any],
                 state_packer: LabeledDataPacker, title=None, show_legend: bool = True,
                 plot_training_trajectories: bool = False,
                 detail_level: DetailLevel = DetailLevel.NO_DETAILS):
        self._state_packer = state_packer
        self._tracking_point_labels = tracking_point_labels
        self._show_cost = detail_level.value >= DetailLevel.COST.value
        self._show_policy_details = detail_level == DetailLevel.COST_AND_POLICY
        self._show_legend = show_legend
        self._title = title
        self._figure = figure
        self._plot_training_trajectories = plot_training_trajectories

        self._gs = gridspec.GridSpecFromSubplotSpec(4 if self._show_cost else 3, 2 if self._show_policy_details else 1,
                                                    subplot_spec=subplot_spec)

        self._axis_sample: Axes3D = figure.add_subplot(self._gs[-3:, 0], projection="3d")
        if self._show_cost:
            self._axis_cost: Axes = figure.add_subplot(self._gs[0, 0])
            # Connect mouse click event
            figure.canvas.mpl_connect("button_press_event", self._on_click)
            figure.canvas.mpl_connect("key_press_event", self._on_key_pressed)
        else:
            self._axis_cost: Axes = None

        if self._show_policy_details:
            self._axis_policy: Axes = figure.add_subplot(self._gs[:, 1])
            self._axis_policy.get_xaxis().set_visible(False)
            self._axis_policy.get_yaxis().set_visible(False)
            self._axis_policy.spines['bottom'].set_visible(False)
            self._axis_policy.spines['top'].set_visible(False)
            self._axis_policy.spines['right'].set_visible(False)
            self._axis_policy.spines['left'].set_visible(False)
            self._axis_policy.set_axis_off()
        else:
            self._axis_policy: Axes = None

        self._training_trajectory: Trajectory = None
        self._test_trajectory: Trajectory = None
        self._distrs: Dict[Any, Tuple[np.ndarray, np.ndarray]] = None
        self._lims3d = None
        self._lims_cost = None
        self._selected_timestep = None
        self._max_timestep = None
        self._highlight_plots = []
        self._tm = TexManager()

    def update_trajectories(self, training_trajectory: Optional[Trajectory] = None,
                            test_trajectory: Optional[Trajectory] = None,
                            policy_trajectory: Optional[Trajectory] = None,
                            refresh_plots: bool = True,
                            training_sample_colors: Optional[List[Any]] = None,
                            test_sample_colors: Optional[List[Any]] = None,
                            policy_trajectory_colors: Optional[List[Any]] = None):
        self._training_trajectory = training_trajectory
        self._test_trajectory = test_trajectory
        self._policy_trajectory = policy_trajectory
        self._training_sample_colors = training_sample_colors
        self._test_sample_colors = test_sample_colors
        self._policy_trajectory_colors = policy_trajectory_colors

        lims = np.array([[np.inf] * 3, [-np.inf] * 3])
        self._distrs = {}

        for label in self._tracking_point_labels:
            all_points = [sample.full_state[label] for trajectory in
                          [training_trajectory, test_trajectory, policy_trajectory] if
                          trajectory is not None for sample in trajectory.samples]

            if training_trajectory is not None and training_trajectory.expected_mu is not None and \
                    label in self._state_packer.labels:
                # Compute gaussian trajectories
                self._distrs[label] = self._compute_trajectory_distribution(training_trajectory, label)
                all_points.append(self._distrs[label][0])

            local_lims = np.array(self._calculate_3d_axis_limits(all_points)).T
            lims[0] = np.minimum(lims[0], local_lims[0])
            lims[1] = np.maximum(lims[1], local_lims[1])

        max_diff = max(lims[1] - lims[0])
        mid = (lims[1] + lims[0]) / 2.0
        self._lims3d = np.array([mid - max_diff / 2.0, mid + max_diff / 2.0]).T

        trajs = [test_trajectory, policy_trajectory]
        if self._plot_training_trajectories:
            trajs.append(training_trajectory)
        self._lims_cost = np.array([np.infty, -np.infty])
        for traj in trajs:
            if traj is not None:
                all_cost = [c.l for c in traj.cost]
                c_min = np.min(all_cost)
                c_max = np.max(all_cost)
                diff = c_max - c_min
                local_lim = np.array([c_min - diff * 0.02, c_max + diff * 0.02])
                self._lims_cost[0] = min(self._lims_cost[0], local_lim[0])
                self._lims_cost[1] = max(self._lims_cost[1], local_lim[1])

        if refresh_plots:
            self.refresh_plots()

    def refresh_plots(self):
        if len(self._tracking_point_labels) > 0:
            # Preserve highlight
            highlight = self._selected_timestep
            self.clear_selected_timestep()
            self._axis_sample.clear()
            if self._show_cost:
                self._axis_cost.clear()
                self._axis_cost.set_ylim(self._lims_cost)
                if highlight is not None:
                    self.select_timestep(highlight)
            self._axis_sample.set_xlim(self._lims3d[0])
            self._axis_sample.set_ylim(self._lims3d[1])
            self._axis_sample.set_zlim(self._lims3d[2])

            if self._title is not None:
                self._axis_sample.set_title(self._title)

            self._axis_sample.tick_params(pad=0)
            self._axis_sample.locator_params(nbins=5)
            for item in itertools.chain(self._axis_sample.get_xticklabels(), self._axis_sample.get_yticklabels(),
                                        self._axis_sample.get_zticklabels()):
                item.set_fontsize(10)
            self._axis_sample.title.set_fontsize(10)

            training_colors = {}
            test_colors = {}
            policy_colors = {}

            # Plot samples
            for label in self._tracking_point_labels:
                if self._training_trajectory is not None:
                    # Training trajectory plotting
                    if self._plot_training_trajectories:
                        for i, sample in enumerate(self._training_trajectory.samples):
                            trajectory_points = sample.full_state[label]
                            if i not in training_colors:
                                color = None if self._training_sample_colors is None else \
                                    self._training_sample_colors[i]
                            else:
                                color = training_colors[i]
                            p = self._plot_3d_points(self._axis_sample, trajectory_points, color=color,
                                                     label="Trajectory sample {}".format(i))[0]
                            training_colors[i] = p.get_color()
                    if label in self._distrs:
                        mu, sigma = self._distrs[label]
                        self._plot_3d_gaussian(self._axis_sample, mu, sigma, edges=100, linestyle="-", linewidth=1.0,
                                               color="red", alpha=0.15, label="LG Controller Distributions")
                        self._plot_3d_points(self._axis_sample, points=mu, linestyle="None", marker="x", markersize=5.0,
                                             markeredgewidth=1.0, color=(0.5, 0, 0), alpha=1.0,
                                             label="LG Controller Means")

                if self._test_trajectory is not None:
                    # Test trajectory plotting
                    for i, sample in enumerate(self._test_trajectory.samples):
                        trajectory_points = sample.full_state[label]
                        if i not in test_colors:
                            color = None if self._test_sample_colors is None else self._test_sample_colors[i]
                        else:
                            color = test_colors[i]
                        p = self._plot_3d_points(self._axis_sample, trajectory_points, color=color,
                                                 label="Test sample {}".format(i))[0]
                        test_colors[i] = p.get_color()

                if self._policy_trajectory is not None:
                    # Policy trajectory plotting
                    for i, sample in enumerate(self._policy_trajectory.samples):
                        trajectory_points = sample.full_state[label]
                        if i not in policy_colors:
                            color = None if self._policy_trajectory_colors is None else \
                                self._policy_trajectory_colors[i]
                        else:
                            color = policy_colors[i]
                        p = self._plot_3d_points(self._axis_sample, trajectory_points, color=color,
                                                 label="Policy sample {}".format(i))[0]
                        policy_colors[i] = p.get_color()

            if self._show_legend:
                handles, labels = self._axis_sample.get_legend_handles_labels()
                filtered_handles = []
                filtered_labels = []
                for handle, label in zip(handles, labels):
                    if label not in filtered_labels:
                        filtered_handles.append(handle)
                        filtered_labels.append(label)
                self._axis_sample.legend(filtered_handles, filtered_labels)
                if self._show_cost:
                    self._axis_cost.legend(filtered_handles[2:], ["Local", "Global"])

            self._max_timestep = 0
            if self._show_cost:
                for t, c in [(self._test_trajectory, test_colors),
                             (self._policy_trajectory, policy_colors)] + (
                                    [(self._training_trajectory,
                                      training_colors)] if self._plot_training_trajectories else []):
                    if t is not None:
                        for i, sample in enumerate(t.samples):
                            cost = t.cost[i].l
                            ts = range(len(cost))
                            self._max_timestep = max(self._max_timestep, len(cost))
                            self._axis_cost.plot(ts, cost, color=c[i])

    @property
    def lims3d(self) -> np.ndarray:
        """
        Limits for 3d plot.
        :return:
        """
        return self._lims3d

    @lims3d.setter
    def lims3d(self, value: np.ndarray):
        """
        Sets the limits for the 3d plot.
        :param value: numpy array containing the limits: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        :return:
        """
        self._lims3d = value

    @property
    def lims_cost(self) -> np.ndarray:
        """
        Limits for the cost plot.
        :return:
        """
        return self._lims_cost

    @lims_cost.setter
    def lims_cost(self, value: np.ndarray):
        """
        Sets the limits for the cost plot.
        :param value: numpy array containing the limits: [y_min, y_max]
        :return:
        """
        self._lims_cost = value

    def _plot_3d_points(self, axis: Axes3D, points: np.ndarray, linestyle: str = '-', linewidth: float = 1.0,
                        marker: Optional[str] = None, markersize: float = 5.0, markeredgewidth: float = 1.0,
                        color: Union[str, Tuple[float, ...]] = 'black', alpha: float = 1.0, label: str = ''):
        lims = [axis.get_xlim(), axis.get_ylim(), axis.get_zlim()]
        data = [points[:, i].copy() for i in range(len(lims))]
        for i in range(len(data)):
            data[i][np.any(np.c_[data[i] < lims[i][0], data[i] > lims[i][1]], axis=1)] = np.nan
        return axis.plot(*data, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                         markeredgewidth=markeredgewidth, color=color, alpha=alpha, label=label)

    def _plot_3d_gaussian(self, axis: Axes3D, mu, sigma, edges=100, linestyle='-.',
                          linewidth=1.0, color='black', alpha=0.1, label=''):
        """
        Plots ellipses in the xy plane representing the Gaussian distributions 
        specified by mu and sigma.
        Args:
            mu    - Tx3 mean vector for (x, y, z)
            sigma - Tx3x3 covariance matrix for (x, y, z)
            edges - the number of edges to use to construct each ellipse
        """
        p = np.linspace(0, 2 * np.pi, edges)
        xy_ellipse = np.c_[np.cos(p), np.sin(p)]
        T = mu.shape[0]

        sigma_xy = sigma[:, 0:2, 0:2]
        u, s, v = np.linalg.svd(sigma_xy)

        for t in range(T):
            xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
            xyz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(
                np.sqrt(s[t, :])), u[t, :, :].T))
            self._plot_3d_points(axis, xyz, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha,
                                 label=label)

    def _calculate_3d_axis_limits(self, data3d: List[np.ndarray]) -> Tuple[Tuple[float, float], ...]:
        min_xyz = np.inf * np.ones((3))
        max_xyz = -np.inf * np.ones((3))
        for data in data3d:
            min_xyz = np.amin(np.concatenate((data, min_xyz.reshape((1, 3)))), axis=0)
            max_xyz = np.amax(np.concatenate((data, max_xyz.reshape((1, 3)))), axis=0)
        xlim = buffered_axis_limits(min_xyz[0], max_xyz[0], buffer_factor=1.25)
        ylim = buffered_axis_limits(min_xyz[1], max_xyz[1], buffer_factor=1.25)
        zlim = buffered_axis_limits(min_xyz[2], max_xyz[2], buffer_factor=1.25)
        return xlim, ylim, zlim

    def _compute_trajectory_distribution(self, trajectory: Trajectory, tracking_point_label: Any) \
            -> Tuple[np.ndarray, np.ndarray]:
        full_mu = trajectory.expected_mu
        full_sigma = trajectory.expected_sigma
        s = self._state_packer.label_slices[tracking_point_label]
        mu = full_mu[:, s]
        sigma = full_sigma[:, s, s]
        return mu, sigma

    def _on_key_pressed(self, event):
        increment = 0

        if event.key == "up":
            increment = 1
        elif event.key == "down":
            increment = -1
        if self._selected_timestep is not None:
            self.select_timestep((self._selected_timestep + increment) % self._max_timestep)
            self._figure.canvas.draw()

    def _on_click(self, event):
        if event.inaxes == self._axis_cost.axes:
            ts = int(event.xdata + 0.5)
            self.select_timestep(ts)
            self._figure.canvas.draw()

    def _mat_to_tex(self, mat: np.ndarray) -> str:
        if len(mat.shape) == 1:
            mat = mat.reshape((1, -1))
        content = r" \\ ".join([" & ".join(["{:0.3f}".format(e) for e in l]) for l in mat])
        return r"\begin{{pmatrix}} {} \end{{pmatrix}}".format(content)

    def _update_policy_details(self):
        if self._training_trajectory is not None:
            ts = self._selected_timestep
            distr: LinearGaussianController = self._training_trajectory.controller
            dyn = self._training_trajectory.dynamics
            sample = self._training_trajectory.samples[0]
            x = sample.states[ts].reshape((-1, 1))
            u = sample.actions[ts].reshape((-1, 1))
            p = plt.rcParams['text.latex.preamble']
            plt.rcParams['text.latex.preamble'] = r"""\usepackage{amsmath} \setcounter{MaxMatrixCols}{50}"""
            lbl_desc = "; ".join(["{}: {}".format(l, d) for l, d in sample.state_packer.label_dimensions.items()])
            tex_code = r"""\begin{{verbatim}} {lbl_desc} \end{{verbatim}}
             Values inserted for x and u are taken from sample 0.
             $u={K} * {x} + {k}={u}$ \\\\
             $d=N({fm} * {xu} + {fv}, {cov})$""".format(
                lbl_desc=lbl_desc,
                u=self._mat_to_tex(u),
                K=self._mat_to_tex(distr.K[ts]),
                x=self._mat_to_tex(x),
                k=self._mat_to_tex(distr.k[ts].reshape((-1, 1))),
                fm=self._mat_to_tex(dyn.Fm[ts]),
                xu=self._mat_to_tex(np.concatenate((x, u), axis=0)),
                fv=self._mat_to_tex(dyn.fv[ts].reshape((-1, 1))),
                cov=self._mat_to_tex(dyn.dyn_covar[ts]))
            self._axis_policy.imshow(self._tm.get_rgba(tex_code))
            plt.rcParams['text.latex.preamble'] = p

    def select_timestep(self, timestep: int):
        """
        Highlight a certain timestep.
        :param timestep: Time step to highlight
        :return:
        """
        self.clear_selected_timestep()
        timestep = max(0, min(self._max_timestep, timestep))
        self._selected_timestep = timestep
        self._highlight_plots.append(self._axis_cost.axvline(x=timestep, color="purple", linewidth=1))
        samples = []
        if self._test_trajectory is not None:
            samples += self._test_trajectory.samples
        if self._training_trajectory is not None and self._plot_training_trajectories:
            samples += self._training_trajectory.samples
        for sample in samples:
            for label in self._tracking_point_labels:
                self._highlight_plots.append(
                    self._axis_sample.plot(*sample.full_state[label][timestep].reshape((3, 1)), color="purple",
                                           marker="o")[0])
        if self._show_policy_details:
            self._update_policy_details()

    def clear_selected_timestep(self):
        """
        Clears the highlight
        :return:
        """
        for plot in self._highlight_plots:
            plot.remove()
        self._highlight_plots = []
        self._selected_timestep = None
        if self._show_policy_details:
            self._axis_policy.clear()
