import logging

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


def Graph(name):
    return globals()["g_" + name]


class MatplotlibTwoPlot:
    def __init__(self):
        plt.switch_backend('Agg')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True)
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])))

        self.ax1.set_xlabel("Iteration")
        self.ax1.set_ylabel("Log-likelihood")
        # self.ax1.set_xticks(np.arange(0, 10, 1))
        for l in self.ax1.get_yticklabels():
            l.set_rotation(45)

        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Gradient squared norm")
        # self.ax2.set_xticks(np.arange(0, 10, 1))
        for l in self.ax2.get_yticklabels():
            l.set_rotation(45)
        # self.ax2.set_ylim(bottom=0)

        # data
        self.x = []     # Note: this could be removed by using np.append(line.get_xdata(), self.x)
        self.y1 = []
        self.y2 = []

        self.lines1 = []
        self.lines2 = []

    def plot(self):
        assert len(self.lines1) > 0

        # plot y1
        if self.lines1[-1] is None:
            assert len(self.y1[-1]) == len(self.x)
            self.lines1[-1] = self.ax1.plot(self.x, self.y1[-1], label=str(len(self.lines1)))[0]
        else:
            assert self.lines1[-1] is not None
            self.lines1[-1].set_xdata(self.x)
            self.lines1[-1].set_ydata(self.y1[-1])
            assert len(self.lines1[-1].get_xdata()) == len(self.lines1[-1].get_ydata())

        self.ax1.legend()

        # plot y2
        if len(self.y2[-1]) > 0:
            for key, value in self.y2[-1].items():

                if key not in self.lines2[-1]:
                    self.lines2[-1][key] = self.ax2.plot(self.x, value, label=str(len(self.lines2)) + '_' + key)[0]
                else:
                    self.lines2[-1][key].set_xdata(self.x)
                    self.lines2[-1][key].set_ydata(value)
                    assert len(self.lines2[-1][key].get_xdata()) == len(self.lines2[-1][key].get_ydata())

            self.ax2.legend()

        # rescale and draw
        # Need both of these in order to rescale
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # We need to draw *and* flush
        self.fig.canvas.draw()
        # self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def prepare_next_run(self):
        self.x.clear()
        self.y1.append([])
        self.y2.append({})

        self.lines1.append(None)
        self.lines2.append({})

    def clear(self):
        self.x.clear()
        self.y1.clear()
        self.y2.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.lines1.clear()
        self.lines2.clear()


class g_estimate_deterministic_atlas(FigureCanvas):

    def __init__(self):
        self.plot = MatplotlibTwoPlot()
        FigureCanvas.__init__(self, self.plot.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def next_run(self):
        """
        Called when the 'run' button is pressed
        """
        self.plot.prepare_next_run()

    def iteration(self, cb_data):
        """
        Callback method called by the Deformetrica API.
        :param cb_data: Current estimator iteration data
        :return: True to continue iterations, False to stop
        """
        current_iteration = cb_data["current_iteration"]
        current_log_likelihood = cb_data["current_log_likelihood"]
        current_gradients = cb_data["gradient"]

        # current_log_likelihood
        # callback can be called several times for the same iteration. Append new x value or not
        if len(self.plot.x) > 0 and self.plot.x[-1] == current_iteration:
            self.plot.x[-1] = current_iteration
            self.plot.y1[-1][-1] = current_log_likelihood
        else:
            self.plot.x.append(current_iteration)
            self.plot.y1[-1].append(current_log_likelihood)

        # current_gradients
        for key in current_gradients.keys():
            grad_square_sum = np.sum(current_gradients[key] ** 2)

            if key not in self.plot.y2[-1]:
                self.plot.y2[-1][key] = []

            if len(self.plot.x) > len(self.plot.y2[-1][key]):
                self.plot.y2[-1][key].append(grad_square_sum)
            else:
                # replace last
                self.plot.y2[-1][key][-1] = grad_square_sum

        self.plot.plot()

        # self.draw()

    def clear(self):
        self.plot.clear()
        self.draw()

    def save(self, url):
        self.plot.fig.savefig(url)
