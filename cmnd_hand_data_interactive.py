import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 8})


class HandDataInteractive:

    def __init__(self, figure_, axes):
        self.figure_ = figure_
        self.world_axis = axes[0]
        self.plane_axes = axes[1:4]
        self.axes = axes[4:7]
        self.labels = ['x', 'y', 'z']
                       # 'd', 'v']
        self.draggable_left = False
        self.draggable_right = False
        self.index_left = 0
        self.index_right = 0
        self.v_lines_left = []
        self.texts_left = []
        self.markers_left = []
        self.v_lines_right = []
        self.texts_right = []
        self.markers_right = []
        self.set_interactivity()

    def set_interactivity(self):
        for i in range(len(self.axes)):
            v_line_left = self.axes[i].axvline(0, color='k', alpha=0.5)
            v_line_left.set_visible(False)
            self.v_lines_left.append(v_line_left)

            text_left = self.axes[i].text(0, 0, '')
            text_left.set_visible(False)
            self.texts_left.append(text_left)

            marker_left, = self.axes[i].plot([0], [0], marker='.', color='crimson', zorder=5)
            marker_left.set_visible(False)
            self.markers_left.append(marker_left)

            v_line_right = self.axes[i].axvline(0, color='k', alpha=0.5)
            v_line_right.set_visible(False)
            self.v_lines_right.append(v_line_right)

            text_right = self.axes[i].text(0, 0, '')
            text_right.set_visible(False)
            self.texts_right.append(text_right)

            marker_right, = self.axes[i].plot([0], [0], marker='.', color='crimson', zorder=5)
            marker_right.set_visible(False)
            self.markers_right.append(marker_right)

            self.axes[i].figure.canvas.mpl_connect("button_press_event", self.on_click)
            self.axes[i].figure.canvas.mpl_connect("button_release_event", self.on_release)
            self.axes[i].figure.canvas.mpl_connect("motion_notify_event", self.on_drag)

    # def run(self):
    #     pass

    def set_initial_conditions(self, original_data, idx_m_onset, idx_m_end):
        self.index_left = idx_m_onset
        self.index_right = idx_m_end + 1

        for i in range(len(self.axes)):
            # LEFT INDEX
            self.v_lines_left[i].set_xdata(original_data['t'][idx_m_onset])
            self.v_lines_left[i].set_visible(True)
            self.texts_left[i].set_position((original_data['t'][idx_m_onset], original_data[self.labels[i]][idx_m_onset]))
            text_ = 't=' + "{:.4f}".format(original_data['t'][idx_m_onset]) + ", " + \
                    self.labels[i] + "=" + "{:.4f}".format(original_data[self.labels[i]][idx_m_onset])
            self.texts_left[i].set_text(text_)
            self.texts_left[i].set_visible(True)
            self.markers_left[i].set_data([original_data['t'][idx_m_onset]], [original_data[self.labels[i]][idx_m_onset]])
            self.markers_left[i].set_visible(True)

            # RIGHT INDEX
            self.v_lines_right[i].set_xdata(original_data['t'][idx_m_end])
            self.v_lines_right[i].set_visible(True)
            self.texts_right[i].set_position((original_data['t'][idx_m_end], original_data[self.labels[i]][idx_m_end]))
            text_ = 't=' + "{:.4f}".format(original_data['t'][idx_m_end]) + ", " + \
                    self.labels[i] + "=" + "{:.4f}".format(original_data[self.labels[i]][idx_m_end])
            self.texts_right[i].set_text(text_)
            self.texts_right[i].set_visible(True)
            self.markers_right[i].set_data([original_data['t'][idx_m_end]], [original_data[self.labels[i]][idx_m_end]])
            self.markers_right[i].set_visible(True)

    # if plt.get_current_fig_manager().toolbar.mode != '': return
    def on_click(self, event):
        if event.inaxes not in self.axes:
            return
        if self.figure_.canvas.toolbar.mode != '':
            return
        if event.button == 1:
            self.draggable_left = True
            self.update_left_visual_information(event)
        elif event.button == 3:
            self.draggable_right = True
            self.update_right_visual_information(event)
        self.figure_.canvas.draw_idle()

    def on_drag(self, event):
        if event.inaxes not in self.axes:
            return
        if self.figure_.canvas.toolbar.mode != '':
            return
        if self.draggable_left:
            self.update_left_visual_information(event)
        if self.draggable_right:
            self.update_right_visual_information(event)

    def on_release(self, event):
        if event.inaxes not in self.axes:
            return
        self.draggable_left = False
        self.draggable_right = False

    def update_left_visual_information(self, event):
        for i in range(len(self.axes)):
            x, y, index = self.get_closest(self.axes[i].lines[0], event.xdata)
            self.v_lines_left[i].set_xdata(x)
            self.texts_left[i].set_position((x, y))
            text_ = 't=' + "{:.4f}".format(x) + ", " + self.labels[i] + "=" + "{:.4f}".format(y)
            self.texts_left[i].set_text(text_)
            self.markers_left[i].set_data([x], [y])
            # Update processed data
            self.index_left = index
            x, y = self.axes[i].lines[0].get_data()
            self.axes[i].lines[1].set_data(x[self.index_left: self.index_right], y[self.index_left: self.index_right])
        self.update_plane_axes(event)
        self.update_world_axis(event)
        self.figure_.canvas.draw_idle()

    def update_right_visual_information(self, event):
        for i in range(len(self.axes)):
            x, y, index = self.get_closest(self.axes[i].lines[0], event.xdata)
            self.v_lines_right[i].set_xdata(x)
            self.texts_right[i].set_position((x, y))
            text_ = 't=' + "{:.4f}".format(x) + ", " + self.labels[i] + "=" + "{:.4f}".format(y)
            self.texts_right[i].set_text(text_)
            self.markers_right[i].set_data([x], [y])
            # Update processed data
            self.index_right = index + 1
            x, y = self.axes[i].lines[0].get_data()
            self.axes[i].lines[1].set_data(x[self.index_left: self.index_right], y[self.index_left: self.index_right])
        self.update_plane_axes(event)
        self.update_world_axis(event)
        self.figure_.canvas.draw_idle()

    @staticmethod
    def get_closest(line, mx):
        x, y = line.get_data()
        idx = np.argmin(np.abs(x - mx))
        return x[idx], y[idx], idx

    def get_indexes(self):
        return self.index_left, self.index_right

    def update_plane_axes(self, event):
        for i in range(len(self.plane_axes)):
            x, y = self.plane_axes[i].lines[0].get_data()
            self.plane_axes[i].lines[1].set_data(x[self.index_left:self.index_right],
                                                 y[self.index_left:self.index_right])

    def update_world_axis(self, event):
        x, y, z = self.world_axis.lines[0].get_data_3d()
        self.world_axis.lines[1].set_data_3d(x[self.index_left:self.index_right],
                                             y[self.index_left:self.index_right],
                                             z[self.index_left:self.index_right])

    def set_new_figure(self, figure_, axes):
        self.figure_ = figure_
        self.world_axis = axes[0]
        self.plane_axes = axes[1:4]
        self.axes = axes[4:9]
        self.draggable_left = False
        self.draggable_right = False
        self.index_left = 0
        self.index_right = 0
        self.v_lines_left = []
        self.texts_left = []
        self.markers_left = []
        self.v_lines_right = []
        self.texts_right = []
        self.markers_right = []
        self.set_interactivity()
