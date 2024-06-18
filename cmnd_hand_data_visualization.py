import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.widgets import Button
import re

from cmnd_hand_data_interactive import HandDataInteractive
from cmnd_data_handler import HandDataHandler
from cmnd_plot_utils import SubplotAllocation, get_sphere
from cmnd_preprocessing_methods import PreprocessingHandData
from cmnd_log_file import HandDataLogFile
from cmnd_config_file import HandDataConfigFile


class HandDataVisualization:

    def __init__(self):

        self.data_handler = HandDataHandler()

        self.log_file = HandDataLogFile()

        self.config_file = HandDataConfigFile()
        self.config_file.create_config_data_file()

        self.preprocessing = PreprocessingHandData()
        self.preprocessing.calculate_mean_sample_frequency()

        self.trial_number = 1
        self.max_trial_number = self.data_handler.n_files
        self.processed_trial = None
        self.temp_processed_trial = None
        self.idx_m_onset = 0
        self.idx_m_end = None
        self.saved_trial = np.full(self.data_handler.n_files, False)

        if self.config_file.last_configuration:
            data = self.config_file.get_data_configuration()
            self.idx_onset_list = data['idx_1'].to_numpy()
            self.idx_end_list = data['idx_2'].to_numpy()
            self.dropped_trial = data['dropped_trial'].to_numpy()
        else:
            self.idx_onset_list = np.full(self.data_handler.n_files, -1)
            self.idx_end_list = np.full(self.data_handler.n_files, -1)
            self.dropped_trial = np.full(self.data_handler.n_files, False)

        self.allocation = SubplotAllocation()
        self.fig_tracker_1 = None
        self.n_rows = 0
        self.n_cols = 0
        self.gs = None
        self.axes_tracker_1 = None
        self.t_limits = ""
        self.x_limits = ""
        self.y_limits = ""
        self.z_limits = ""
        self.x_label_list = ['x', 'x', 'z',
                             't', 't', 't',
                             't', 't']
        self.y_label_list = ['y', 'z', 'y',
                             'x', 'y', 'z',
                             'd', 'v']
        plt.ion()
        self.fig_tracker_1 = plt.figure(0, figsize=(16, 8))
        self.set_layout_tracker_1()
        self.rows = [True, True, False, False, False]
        self.cols = [True, True, True, False]
        self.update_layout(self.rows, self.cols)
        self.interactive = HandDataInteractive(self.fig_tracker_1, self.axes_tracker_1)

        self.initiation_time_method = None
        self.it_threshold = 0

        self.play_show = False
        self.stop_show = True
        self.sleep_time = 0.1
        self.zoom_ = False

        self.stimuli_list = [False for _ in range(5)]
        self.targets_list = [False for _ in range(5)]
        self.stimuli_patches = [[None for _ in range(5)] for _ in range(3)]
        self.targets_patches = [[None for _ in range(5)] for _ in range(3)]
        self.stimuli_surfaces = [None for _ in range(5)]
        self.targets_surfaces = [None for _ in range(5)]

    def plot_trial(self):
        # If the user closed the matplotlib window
        if not plt.fignum_exists(0):

            plt.ion()
            self.fig_tracker_1 = plt.figure(0, figsize=(16, 8))
            self.set_layout_tracker_1()
            self.update_layout(self.rows, self.cols)
            self.interactive.set_new_figure(self.fig_tracker_1, self.axes_tracker_1)

        # Update the subtitle of the figure that indicates if the current trial has been dropped
        if self.dropped_trial[self.trial_number - 1]:
            self.fig_tracker_1.suptitle("DROPPED TRIAL", color='red', fontsize=15)
        else:
            self.fig_tracker_1.suptitle("")

        self.fig_tracker_1.canvas.manager.set_window_title("Trial " + str(self.trial_number))
        self.show_mean_sample_frequency()

        self.temp_processed_trial = self.data_handler.get_trial_data_hand_tracker(self.trial_number)
        self.processed_trial = self.data_handler.get_trial_data_reach_trajectories(self.trial_number)

        # Original Data (References)
        self.plot_data(0, self.temp_processed_trial, 0, None, 'blue', 0.1)

        # Movement onset index
        self.get_movement_onset(self.temp_processed_trial)
        self.get_movement_end(self.temp_processed_trial)

        # Processed Data (Reference)
        if self.saved_trial[self.trial_number - 1]:
            self.plot_data(1, self.processed_trial, 0, None, 'b', 1.0)
        else:
            self.plot_data(1, self.processed_trial, self.idx_m_onset, self.idx_m_end + 1, 'b', 1.0)

        self.interactive.set_initial_conditions(self.temp_processed_trial, self.idx_m_onset, self.idx_m_end)

        for ax in self.fig_tracker_1.get_axes()[0:9]:
            ax.relim()
            ax.autoscale()
        # self.relim_autoscale_3d()
        self.apply_limits(self.t_limits, self.x_limits, self.y_limits, self.z_limits)
        self.fig_tracker_1.canvas.draw()

    def show_mean_sample_frequency(self):
        sf_mean = self.log_file.get_trial_data_from_log('Mean Sample Frequency', self.trial_number)
        self.axes_tracker_1[0].set_title(str(sf_mean) + " Hz", fontsize=15)

    def relim_autoscale_3d(self):
        axis = self.fig_tracker_1.get_axes()[0]
        x, y, z = axis.lines[0].get_data_3d()
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_zlim(z_min, z_max)

    def plot_data(self, line_index, data, index_i, index_f, color, alpha):
        # ---- 3D WORLD ---- #
        self.plot_3d_time_series(line_index, data['x'], data['z'], data['y'], index_i, index_f, color, alpha)

        # ---- PLANES ---- #
        # X - Y PLANE
        self.plot_time_series(1, line_index, data['x'], data['y'], index_i, index_f, color, alpha)
        # X - Z PLANE
        self.plot_time_series(2, line_index, data['x'], data['z'], index_i, index_f, color, alpha)
        # Z - Y PLANE
        self.plot_time_series(3, line_index, data['z'], data['y'], index_i, index_f, color, alpha)

        # ---- AXES ---- #
        # T VS X
        self.plot_time_series(4, line_index, data['t'], data['x'], index_i, index_f, color, alpha)
        # T VS Y
        self.plot_time_series(5, line_index, data['t'], data['y'], index_i, index_f, color, alpha)
        # T VS Z
        self.plot_time_series(6, line_index, data['t'], data['z'], index_i, index_f, color, alpha)

        # ---- DERIVATIVES ---- #
        # D (0th derivative)
        self.plot_time_series(7, line_index, data['t'], data['d'], index_i, index_f, color, alpha)
        # V (1st derivative)
        self.plot_time_series(8, line_index, data['t'], data['v'], index_i, index_f, color, alpha)

    def plot_time_series(self, axes_index, line_index, series_1, series_2, index_i, index_f, color, alpha):
        self.axes_tracker_1[axes_index].lines[line_index].set_data(series_1.iloc[index_i:index_f], series_2.iloc[index_i:index_f])
        self.axes_tracker_1[axes_index].lines[line_index].set_color(color)
        self.axes_tracker_1[axes_index].lines[line_index].set_alpha(alpha)

    def plot_3d_time_series(self, line_index, series_1, series_2, series_3, index_i, index_f, color, alpha):
        self.axes_tracker_1[0].lines[line_index].set_data_3d(series_1.iloc[index_i:index_f],
                                                             series_2.iloc[index_i:index_f],
                                                             series_3.iloc[index_i:index_f])
        self.axes_tracker_1[0].lines[line_index].set_color(color)
        self.axes_tracker_1[0].lines[line_index].set_alpha(alpha)

    def get_movement_end(self, trial_data):
        if self.idx_end_list[self.trial_number - 1] != -1:
            self.idx_m_end = self.idx_end_list[self.trial_number - 1]
            # print("End from List", self.idx_m_end)
        else:
            self.idx_m_end = trial_data['t'].index[trial_data['t'].index[-1]]
            # print("End from Default", self.idx_m_end)

    def get_movement_onset(self, trial_data):
        if self.idx_onset_list[self.trial_number - 1] != -1:
            self.idx_m_onset = self.idx_onset_list[self.trial_number - 1]
            # self.initiation_time = trial_data['t'].iloc[self.idx_m_onset]
            # print("Onset from List", self.idx_m_onset)

        elif self.initiation_time_method == "Spatial":
            idx_, it_ = self.preprocessing.initiation_time_spatial_condition(trial_data['t'].to_numpy(),
                                                                             self.trial_number)
            self.idx_m_onset = idx_
            # self.initiation_time = it_
            # print("Onset from Spatial method", self.idx_m_onset)

        elif self.initiation_time_method == "Speed":
            idx_, it_ = self.preprocessing.initiation_time_velocity_condition(trial_data['t'].to_numpy(),
                                                                              trial_data['v'].to_numpy(),
                                                                              self.it_threshold)
            self.idx_m_onset = idx_
            # self.initiation_time = it_
            # print("Onset from Speed Method", self.idx_m_onset)

        else:
            self.idx_m_onset = 0
            # self.initiation_time = trial_data['t'].iloc[self.idx_m_onset]
            # print("Onset from Default", self.idx_m_onset)

    def set_initiation_time_method(self, initiation_time):
        self.initiation_time_method = initiation_time

    def set_initiation_time_threshold(self, threshold):
        self.it_threshold = threshold

    def keyboard_control(self, event):
        if event.key == "right":
            self.plot_next(None)
        elif event.key == "left":
            self.plot_previous(None)
        elif event.key == "down":
            self.play_trials(None)

    def play_trials(self, event):
        if not self.play_show:
            self.play_show = True
        else:
            self.play_show = False
        while self.play_show:
            self.plot_next(event)
            plt.pause(self.sleep_time)

    def reset_trial(self, event):
        self.idx_onset_list[self.trial_number - 1] = -1
        self.idx_end_list[self.trial_number - 1] = -1
        self.saved_trial[self.trial_number - 1] = False
        preprocessed_trial = self.data_handler.get_trial_data_hand_tracker(self.trial_number)
        self.data_handler.save_trial_data_reach_trajectories(self.trial_number, preprocessed_trial)
        self.plot_trial()

    def save_trial(self):
        if not self.saved_trial[self.trial_number - 1]:
            self.saved_trial[self.trial_number - 1] = True

            lb, ub = self.interactive.get_indexes()

            if self.idx_m_onset == lb:
                self.idx_onset_list[self.trial_number - 1] = self.idx_m_onset
            else:
                self.idx_onset_list[self.trial_number - 1] = lb
                # self.initiation_time = self.temp_processed_trial['t'].iloc[lb]

            if self.idx_m_end == ub - 1:
                self.idx_end_list[self.trial_number - 1] = self.idx_m_end
            else:
                self.idx_end_list[self.trial_number - 1] = ub - 1

            idx_1 = self.idx_onset_list[self.trial_number - 1]
            idx_2 = self.idx_end_list[self.trial_number - 1]
            processed_trial = self.temp_processed_trial.iloc[idx_1:idx_2 + 1].copy()

            initiation_time = processed_trial['t'].iloc[0]
            end_time = processed_trial['t'].iloc[-1]
            movement_time = end_time - initiation_time

            # ---- SAVE DATA ---- #
            print("SAVING DATA OF TRIAL", self.trial_number)
            self.data_handler.save_trial_data_reach_trajectories(self.trial_number, processed_trial)
            results_log = {'Initiation Time': initiation_time,
                           'Movement Time': movement_time,
                           'Total Time': end_time}
            self.log_file.save_trial_to_log(self.trial_number, results_log)
            results_config = {'dropped_trial': int(self.dropped_trial[self.trial_number - 1]),
                              'idx_1': idx_1,
                              'idx_2': idx_2}
            self.config_file.save_trial_to_config(self.trial_number, results_config)

    def save_all_trials(self, event):
        for trial_number in range(1, self.max_trial_number + 1):
            if not self.saved_trial[trial_number - 1]:
                self.saved_trial[trial_number - 1] = True

                idx_1 = self.idx_onset_list[trial_number - 1]
                idx_2 = self.idx_end_list[trial_number - 1]

                temp_processed_trial = self.data_handler.get_trial_data_hand_tracker(trial_number)
                processed_trial = temp_processed_trial.iloc[idx_1:idx_2 + 1].copy()
                initiation_time = processed_trial['t'].iloc[0]
                end_time = processed_trial['t'].iloc[-1]
                movement_time = end_time - initiation_time

                # ---- SAVE DATA ---- #
                self.data_handler.save_trial_data_reach_trajectories(trial_number, processed_trial)
                results_log = {'Initiation Time': initiation_time, 'Movement Time': movement_time, 'Total Time': end_time}
                self.log_file.save_trial_to_log(trial_number, results_log)
                results_config = {'dropped_trial': int(self.dropped_trial[trial_number - 1]), 'idx_1': idx_1, 'idx_2': idx_2}
                self.config_file.save_trial_to_config(trial_number, results_config)

        print("ALL TRIALS HAVE BEEN SAVED")

    def plot_next(self, event):
        self.save_trial()
        self.zoom_ = False
        self.trial_number += 1
        if self.trial_number <= self.max_trial_number:
            self.plot_trial()
        else:
            self.play_show = False
            self.trial_number -= 1

    def plot_previous(self, event):
        self.zoom_ = False
        self.trial_number -= 1
        if self.trial_number >= 1:
            self.plot_trial()
        else:
            self.trial_number += 1

    def update_layout(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.allocation.update_layout(self.fig_tracker_1, rows, cols)

    def set_layout_tracker_1(self):
        gs = gridspec.GridSpec(8, 4)

        # PLANES
        self.fig_tracker_1.add_subplot(gs[0:3, 0], projection='3d')   # 3D PLOT
        self.fig_tracker_1.add_subplot(gs[0:3, 1])                    # XY
        self.fig_tracker_1.add_subplot(gs[0:3, 2])                    # XZ
        self.fig_tracker_1.add_subplot(gs[0:3, 3])                    # ZY

        # T vs X, Y, Z
        self.fig_tracker_1.add_subplot(gs[3, 0:4])                    # X
        ax_shared = self.fig_tracker_1.gca()
        self.fig_tracker_1.add_subplot(gs[4, 0:4], sharex=ax_shared)  # Y
        self.fig_tracker_1.add_subplot(gs[5, 0:4], sharex=ax_shared)  # Z

        # DERIVATIVES
        self.fig_tracker_1.add_subplot(gs[6, 0:4], sharex=ax_shared)  # D
        self.fig_tracker_1.add_subplot(gs[7, 0:4], sharex=ax_shared)  # V

        self.axes_tracker_1 = self.fig_tracker_1.get_axes()

        self.axes_tracker_1[0].plot([], [], [], '*')
        self.axes_tracker_1[0].plot([], [], [], '.')
        self.axes_tracker_1[0].plot([], [], [], '.')
        self.axes_tracker_1[0].set_xlabel('x')
        self.axes_tracker_1[0].set_ylabel('z')
        self.axes_tracker_1[0].set_zlabel('y')
        ax = self.axes_tracker_1[0]
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')], zoom=1.1)

        for i, ax_i in enumerate(self.axes_tracker_1[1:]):
            ax_i.grid(True)
            ax_i.plot([], [], '.')
            ax_i.plot([], [], '.')
            ax_i.set_xlabel(self.x_label_list[i])
            ax_i.set_ylabel(self.y_label_list[i])

        self.ax_prev = plt.axes([0.3375, 0.01, 0.075, 0.05])
        self.button_previous = Button(self.ax_prev, "Previous")
        self.button_previous.on_clicked(self.plot_previous)

        self.ax_reset = plt.axes([0.4375, 0.01, 0.075, 0.05])
        self.button_reset = Button(self.ax_reset, "Reset")
        self.button_reset.on_clicked(self.reset_trial)

        self.ax_next = plt.axes([0.6375, 0.01, 0.075, 0.05])
        self.button_next = Button(self.ax_next, "Next")
        self.button_next.on_clicked(self.plot_next)

        self.axes_zoom = plt.axes([0.5375, 0.01, 0.075, 0.05])
        self.button_zoom = Button(self.axes_zoom, 'Zoom')
        self.button_zoom.on_clicked(self.plot_zoom)

        if self.config_file.last_configuration:
            self.axes_save_all = plt.axes([0.8775, 0.01, 0.075, 0.05])
            self.button_save_all = Button(self.axes_save_all, 'Save All')
            self.button_save_all.on_clicked(self.save_all_trials)

        plt.connect('key_press_event', self.keyboard_control)

        self.fig_tracker_1.subplots_adjust(left=0.05, top=0.95, right=0.95, bottom=0.1, hspace=0.5, wspace=0.25)

    def plot_zoom(self, event):
        # 2D PLOT
        if not self.zoom_:
            self.x_lim, self.y_lim = [], []
            for i in range(1, 4):
                axis = self.fig_tracker_1.get_axes()[i]
                x_lim = axis.get_xlim()
                y_lim = axis.get_ylim()
                self.x_lim.append(x_lim)
                self.y_lim.append(y_lim)
                x, y = axis.lines[0].get_data()
                x = x[0:self.idx_m_onset]
                y = y[0:self.idx_m_onset]
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                axis.set_xlim(x_min, x_max)
                axis.set_ylim(y_min, y_max)
        else:
            for i in range(1, 4):
                axis = self.fig_tracker_1.get_axes()[i]
                axis.set_xlim(self.x_lim[i-1][0], self.x_lim[i-1][1])
                axis.set_ylim(self.y_lim[i-1][0], self.y_lim[i-1][1])

        # 3D PLOT
        if not self.zoom_:
            self.x_3dlim, self.y_3dlim, self.z_3dlim = [], [], []
            axis = self.fig_tracker_1.get_axes()[0]
            x_lim = axis.get_xlim()
            y_lim = axis.get_ylim()
            z_lim = axis.get_zlim()
            self.x_3dlim.append(x_lim)
            self.y_3dlim.append(y_lim)
            self.z_3dlim.append(z_lim)
            x, y, z = axis.lines[0].get_data_3d()
            x = x[0:self.idx_m_onset]
            y = y[0:self.idx_m_onset]
            z = z[0:self.idx_m_onset]
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min, z_max = np.min(z), np.max(z)
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_zlim(z_min, z_max)
            self.zoom_ = True
        else:
            axis = self.fig_tracker_1.get_axes()[0]
            axis.set_xlim(self.x_3dlim[0][0], self.x_3dlim[0][1])
            axis.set_ylim(self.y_3dlim[0][0], self.y_3dlim[0][1])
            axis.set_zlim(self.z_3dlim[0][0], self.z_3dlim[0][1])
            self.zoom_ = False

    def set_limits(self, t_limits, x_limits, y_limits, z_limits):
        self.t_limits = t_limits
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.apply_limits(self.t_limits, self.x_limits, self.y_limits, self.z_limits)

    def apply_limits(self, t_limits, x_limits, y_limits, z_limits):
        t_limits = re.findall(r'[-+]?\d+(?:\.\d+)?', t_limits)
        x_limits = re.findall(r'[-+]?\d+(?:\.\d+)?', x_limits)
        y_limits = re.findall(r'[-+]?\d+(?:\.\d+)?', y_limits)
        z_limits = re.findall(r'[-+]?\d+(?:\.\d+)?', z_limits)

        axes = self.fig_tracker_1.get_axes()[1:]
        for ax in axes:
            if ax.get_xlabel() == 't' and len(t_limits) != 0:
                ax.set_xlim(float(t_limits[0]), float(t_limits[1]))
            elif ax.get_xlabel() == 'x' and len(x_limits) != 0:
                ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
            elif ax.get_xlabel() == 'y' and len(y_limits) != 0:
                ax.set_xlim(float(y_limits[0]), float(y_limits[1]))
            elif ax.get_xlabel() == 'z' and len(z_limits) != 0:
                ax.set_xlim(float(z_limits[0]), float(z_limits[1]))

            if ax.get_ylabel() == 'x' and len(x_limits) != 0:
                ax.set_ylim(float(x_limits[0]), float(x_limits[1]))
            elif ax.get_ylabel() == 'y' and len(y_limits) != 0:
                ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
            elif ax.get_ylabel() == 'z' and len(z_limits) != 0:
                ax.set_ylim(float(z_limits[0]), float(z_limits[1]))

        ax = self.fig_tracker_1.get_axes()[0]
        if ax.get_xlabel() == 'x' and len(x_limits) != 0:
            ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
        if ax.get_ylabel() == 'z' and len(z_limits) != 0:
            ax.set_ylim(float(z_limits[0]), float(z_limits[1]))
        if ax.get_zlabel() == 'y' and len(y_limits) != 0:
            ax.set_zlim(float(y_limits[0]), float(y_limits[1]))

        self.fig_tracker_1.canvas.draw_idle()

    def draw_stimulus(self, number, pos, size):
        if not self.stimuli_list[number - 1]:

            # 2D Stimuli
            axes = self.fig_tracker_1.get_axes()[1:4]
            for a, ax in enumerate(axes):
                if a == 0:
                    i, j = 0, 1
                elif a == 1:
                    i, j = 0, 2
                else:
                    i, j = 2, 1
                stimulus = patches.Circle((float(pos[i]), float(pos[j])), float(size), fill=False)
                p_ = ax.add_patch(stimulus)
                self.stimuli_patches[a][number - 1] = p_
                ax.relim()
                ax.autoscale()
                # self.fig_tracker_1.canvas.draw_idle()

            # 3D Stimuli
            ax = self.fig_tracker_1.get_axes()[0]
            x, y, z = get_sphere(float(pos[0]), float(pos[2]), float(pos[1]), float(size))
            s_ = ax.plot_surface(x, y, z, alpha=0.1, color='grey')
            self.stimuli_surfaces[number - 1] = s_
            ax.relim()
            ax.autoscale()
            # self.fig_tracker_1.canvas.draw_idle()
            self.fig_tracker_1.canvas.draw()

            self.stimuli_list[number - 1] = True

    def draw_target(self, number, pos, size):
        if not self.targets_list[number - 1]:

            # 2D Stimuli
            axes = self.fig_tracker_1.get_axes()[1:4]
            for a, ax in enumerate(axes):
                if a == 0:
                    i, j = 0, 1
                elif a == 1:
                    i, j = 0, 2
                else:
                    i, j = 2, 1
                targets = patches.Circle((float(pos[i]), float(pos[j])), float(size), fill=False)
                p_ = ax.add_patch(targets)
                self.targets_patches[a][number - 1] = p_
                ax.relim()
                ax.autoscale()
                # self.fig_tracker_1.canvas.draw_idle()

            # 3D Stimuli
            ax = self.fig_tracker_1.get_axes()[0]
            x, y, z = get_sphere(float(pos[0]), float(pos[2]), float(pos[1]), float(size))
            s_ = ax.plot_surface(x, y, z, alpha=0.1, color='grey')
            self.targets_surfaces[number - 1] = s_
            ax.relim()
            ax.autoscale()
            # self.fig_tracker_1.canvas.draw_idle()
            self.fig_tracker_1.canvas.draw()

            self.targets_list[number - 1] = True

    def delete_stimulus(self, number):
        if self.stimuli_list[number - 1]:
            # Delete 2D stimuli
            axes = self.fig_tracker_1.get_axes()[1:4]
            for a, ax in enumerate(axes):
                p = self.stimuli_patches[a][number - 1]
                p.remove()
                self.stimuli_patches[a][number - 1] = None

            # Delete 3D stimuli
            s = self.stimuli_surfaces[number - 1]
            s.remove()
            self.stimuli_surfaces[number - 1] = None
            self.stimuli_list[number - 1] = False

    def delete_target(self, number):
        if self.targets_list[number - 1]:
            # Delete 2D stimuli
            axes = self.fig_tracker_1.get_axes()[1:4]
            for a, ax in enumerate(axes):
                p = self.targets_patches[a][number - 1]
                p.remove()
                self.targets_patches[a][number - 1] = None

            # Delete 3D stimuli
            s = self.targets_surfaces[number - 1]
            s.remove()
            self.targets_surfaces[number - 1] = None
            self.targets_list[number - 1] = False

    def delete_objects(self):
        print("Deleting objects")
        for i, stimulus in enumerate(self.stimuli_list, start=1):
            if stimulus:
                self.delete_stimulus(i)

        for i, targets in enumerate(self.targets_list, start=1):
            if targets:
                self.delete_target(i)

    def drop_trial(self, dropped_):
        self.dropped_trial[self.trial_number - 1] = dropped_

    def set_sleep_time(self, sleep_time):
        self.sleep_time = sleep_time

