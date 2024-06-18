'''
Fake class to test visualization and interactivity
Different values are hardcoded (see the folder with the test data)
'''

import numpy as np
import pandas as pd
import os

class HandDataHandler():

    def __init__(self):

        self.data_path = None
        # Real experiments have hundreds of trials
        self.n_files = 10

    def get_trial_data_hand_tracker(self, trial_number):
        trial_file = r"\test_data\P36\S001\resampled_trackers\controllertracker_movement_T" \
                     + str(trial_number).zfill(3) + ".csv"
        trial_path = os.getcwd() + trial_file
        trial_data = pd.read_csv(trial_path, usecols=['t', 'x', 'y', 'z'])
        return trial_data

    # @staticmethod
    # def rename_headers(dataframe):
    #     dataframe.columns = ['t', 'x', 'y', 'z']


if __name__ == "__main__":

    data_handler = HandDataHandler()
    data_ = data_handler.get_trial_data_hand_tracker(1)
    print(data_)