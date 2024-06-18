import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def get_sphere(cx, cy, cz, scale):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return scale * x + cx, scale * y + cy, scale * z + cz


def get_cube(cx, cy, cz, scale):
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)
    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)
    return scale * x + cx, scale * y + cy, scale * z + cz


class SubplotAllocation:

    def __init__(self):
        self.geometry = []

    def get_new_geometry(self, rows_allocation, cols_allocation):
        i = 0
        for allocate in cols_allocation:
            if allocate:
                self.geometry.append((slice(0, 3), i))
                i += 1
            else:
                self.geometry.append([])
        j = 0
        for allocate in rows_allocation:
            if allocate:
                if sum(cols_allocation) > 0:
                    self.geometry.append((3 + j, slice(0, sum(cols_allocation))))
                else:
                    self.geometry.append((j, slice(0, 3)))
                j += 1
            else:
                self.geometry.append([])

    def update_layout(self, figure, rows_allocation: list, cols_allocation: list):
        self.get_new_geometry(rows_allocation, cols_allocation)
        # print("Final Geometry", self.geometry)
        allocation = cols_allocation + rows_allocation

        if sum(cols_allocation) > 0:
            gs = gridspec.GridSpec(3 + sum(rows_allocation), sum(cols_allocation))
        else:
            gs = gridspec.GridSpec(sum(rows_allocation), 3)

        for allocate, ax, geo in zip(allocation, figure.axes, self.geometry):
            if allocate:
                ax.set_visible(True)
                pos = gs[geo].get_position(figure)
                ax.set_position(pos)
                ax.set_subplotspec(gs[geo])
            else:
                ax.set_visible(False)
        self.geometry = []
        figure.canvas.draw_idle()


if __name__ == "__main__":

    figure = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(8, 3)
    ax = figure.add_subplot(gs[0:3, 0])
    ax.grid(True)
    ax.plot([1, 2, 3])
    ax = figure.add_subplot(gs[0:3, 1])
    ax.grid(True)
    ax.plot([3, 2, 1])
    ax = figure.add_subplot(gs[0:3, 2])
    ax.grid(True)
    ax.plot([2, 2, 2])
    ax = figure.add_subplot(gs[3, 0:3])
    ax.grid(True)
    ax.plot([1, 2, 3])
    ax = figure.add_subplot(gs[4, 0:3])
    ax.grid(True)
    ax.plot([3, 2, 1])
    ax = figure.add_subplot(gs[5, 0:3])
    ax.grid(True)
    ax.plot([2, 2, 2])
    ax = figure.add_subplot(gs[6, 0:3])
    ax.grid(True)
    ax.plot([2, 2, 2])
    ax = figure.add_subplot(gs[7, 0:3])
    ax.grid(True)
    ax.plot([2, 2, 2])

    # plt.show()

    allocation = SubplotAllocation()

    cols_allocation = [1, 1, 1]
    rows_allocation = [1, 1, 1, 1, 1]
    allocation.update_layout(figure, rows_allocation, cols_allocation)

    plt.show()