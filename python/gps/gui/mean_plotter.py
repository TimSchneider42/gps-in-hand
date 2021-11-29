"""
Mean Plotter

The Mean Plotter plots data along with its mean. The data is plotted as dots
whereas the mean is a connected line.

This is used to plot the mean cost after each iteration, along with the initial
costs for each sample and condition.
"""
from typing import List

import numpy as np
from matplotlib.axes import Axes

from gps.gui.util import buffered_axis_limits


class MeanPlotter:
    def __init__(self, ax: Axes, labels: List[str], colors=None, alpha: float = 1.0, min_itr: float = 10):
        self._ax = ax

        self._labels = labels
        self._colors = colors

        self._alpha = alpha
        self._min_itr = min_itr

        self._ax.set_xlim(0 - 0.5, self._min_itr + 0.5)
        self._ax.set_ylim(0, 1)
        self._ax.minorticks_on()

        self._plots = []
        self._plots_mean = []

        # Plot labels
        for i, l in enumerate(labels):
            color = None if self._colors is None else self._colors[i]
            plot_mean = self._ax.plot([], [], "-x", markeredgewidth=1.0, color=color, alpha=1.0, label=l)[0]
            self._plots_mean.append(plot_mean)
            self._plots.append(
                self._ax.plot([], [], ".", markersize=4, color=plot_mean.get_color(), alpha=self._alpha)[0])

        self._ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
        self._highlight_plot = None
        self._highlight = None

    def update(self, x: np.ndarray, y: List[np.ndarray]):
        """
        Updates the plot with new data.
        :param x:   1-dimensional array containing the x data
        :param y:   List of 2-dimensional matrices containing the y data. Each list entry contains data of one series,
                    which has the shape (len(x), SAMPLE-COUNT)
        :return:
        """
        x_max = np.amax(x)
        x_min = np.amin(x)

        y_min = float("inf")
        y_max = float("-inf")
        for i, yi in enumerate(y):
            y_min = min(np.amin(yi), y_min)
            y_max = max(np.amax(yi), y_max)

            # Plot mean
            mean = np.mean(yi, axis=1)
            self._plots_mean[i].set_data(x, mean)

            # Plot data
            xi, yi2 = zip(*[(x, yii) for x, yi in zip(x, yi) for yii in yi])
            self._plots[i].set_data(xi, yi2)

        self._ax.set_xlim(x_min - 0.5, max(x_max, self._min_itr) + 0.5)
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.1))

        # Refresh highlight
        if self._highlight is not None:
            self.highlight(self._highlight)

    def highlight(self, x: float):
        """
        Highlight a certain axis section
        :param x: Section to highlight
        :return:
        """
        self.clear_highlight()
        self._highlight = x
        self._highlight_plot = self._ax.axvline(x=x, color="red", linewidth=1)

    def clear_highlight(self):
        """
        Clears the highlight
        :return:
        """
        if self._highlight_plot is not None:
            self._highlight_plot.remove()
            self._highlight = None

    @property
    def axes(self) -> Axes:
        return self._ax
