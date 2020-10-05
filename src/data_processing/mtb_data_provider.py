# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import subprocess
import requests
from xml.etree.ElementTree import fromstring, ElementTree
from geopy.distance import geodesic
import polyline
from pyod.models.knn import KNN
import math
from tqdm import tqdm_notebook as tqdm
from tempfile import TemporaryFile
import os.path
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter


LATITUDE_KEY = 'position_lat'
LONGITUDE_KEY = 'position_long'


class MtbDataProvider:


    def calculate_features(self, samples, step_size, sub_sample_length, feature_thresholds, keep_positions=False):
        print("Calculating Features:", samples.shape, sub_sample_length)

        result = []

        lower_thresholds, upper_thresholds = feature_thresholds

        ACC_X = 0
        ACC_Y = 1
        ACC_Z = 2
        ALTITUDE = 3
        SPEED = 4
        HEART_RATE = 5
        HEADING = 6
        LAT = 7
        LNG = 8

        for i in tqdm(range(len(samples))):
            sample = samples[i]
            sample_slices, _, _ = self.slice_into_windows(sample, window_length=sub_sample_length, step_size=step_size, includes_timestamp=False)

            feature_vector = []

            for data_column in [ACC_X, ACC_Y, ACC_Z, SPEED]:
                # Min, Max, Means
                feature_vector.append(np.median(sample[:, data_column]))


                # Feature Templates: _ points above or below percentile
                feature_vector.append(np.sum(sample[:, data_column] > upper_thresholds[data_column]))
                feature_vector.append(np.sum(sample[:, data_column] < lower_thresholds[data_column]))

            # max altitude change, in sub samples
            altitude_changes = [sample_slices[:, :, ALTITUDE].max(axis=1) - sample_slices[:, :, ALTITUDE].min(axis=1)]
            feature_vector.append(np.max(altitude_changes))

            # max speed change, in sub samples
            speed_changes = [sample_slices[:, :, SPEED].max(axis=1) - sample_slices[:, :, SPEED].min(axis=1)]
            feature_vector.append(np.max(speed_changes))

            # max heading change per distance
            feature_vector.append(self.calc_max_heading_change_per_distance(sample_slices, LAT, LNG, HEADING))

            if keep_positions:
                feature_vector.append(sample[:, LAT].mean())
                feature_vector.append(sample[:, LNG].mean())

            result.append(feature_vector)

        return np.array(result)


    # Find the biggest heading change per smallest distance (radius) within the slices
    def calc_max_heading_change_per_distance(self, sample_slices, LAT, LNG, HEADING):

        max_heading_distance_ratio =  0

        # Iterate through all samples
        for sample_slice in sample_slices:

            # Pick only the heading values
            heading_slice = sample_slice[:, HEADING]

            # create every combination of headings within this sample
            heading_combinations = np.array(np.meshgrid(heading_slice, heading_slice)).T.reshape(-1,2)
            heading_combinations = np.unique(heading_combinations, axis=0)
            # calculate heading delta for every combination
            deltas = np.apply_along_axis(self.calc_heading_delta, 1, heading_combinations)

            # Get the max heading delta
            max_delta = np.max(deltas)
            if max_delta == 0:
                continue

            # Check which combinations led to this max heading change
            max_delta_indices = np.where(deltas == max_delta)
            #print("max_delta_indices", max_delta_indices)
            max_delta_combinations = heading_combinations[max_delta_indices]

            # For each combination leading to a max heading change
            for max_delta_combination in max_delta_combinations:
                # Find the corresponding element within the original heading slice
                sample1_index = np.where(heading_slice == max_delta_combination[0])[0][0]
                sample2_index = np.where(heading_slice == max_delta_combination[1])[0][0]

                # Find the corresponding element within the full samples (not only heading)
                sample1 = sample_slice[sample1_index]
                sample2 = sample_slice[sample2_index]

                # Calculate the distance between those two elements
                location1 = (sample1[LAT], sample1[LNG])
                location2 = (sample2[LAT], sample2[LNG])
                distance = geodesic(location1, location2).meters

                if distance > 0:
                    # Calculate the ratio (how tight a corner is)
                    max_heading_distance_ratio = np.max([max_delta/distance, max_heading_distance_ratio])
                else:
                    max_heading_distance_ratio = 99

        return max_heading_distance_ratio

    # Hadings are given in rad. This calculates the angle between two rad values properly
    def calc_heading_delta(self, heading):
        heading_delta = heading[1] - heading[0]

        while (heading_delta < -math.pi):
            heading_delta += 2*math.pi
        while (heading_delta > math.pi):
            heading_delta -= 2*math.pi

        return np.abs(heading_delta)

    def slice_into_windows(self, data, labels=None, window_length=50, step_size=1, includes_timestamp=True):
        windows = []
        windowed_labels = []

        if labels is None:
            labels = np.zeros((len(data), 1))

        if len(labels.shape) == 1:
            labels = labels.reshape((labels.shape[0], 1))

        stride = int(window_length // (1 // step_size))
        for i in range(0, data.shape[0], stride):
            label = 0
            if i + window_length > data.shape[0]:
                window = data[-window_length:,  :]
                label_slice = []

                for label_set in labels[-window_length:].T:
                    most_common = Counter(label_set).most_common()[0][0]
                    label_slice.append(most_common)
            else:
                window = data[i:i + window_length, :]
                label_slice = []

                for label_set in labels[i:i + window_length].T:
                    most_common = Counter(label_set).most_common()[0][0]
                    label_slice.append(most_common)

            windows.append(window)
            windowed_labels.append(label_slice)

        windows = np.array(windows)
        timestamps = windows[:, 0, 0] # seperate timestamps from data

        if includes_timestamp:
            windows = windows[:, :, 1:]

        return windows, np.array(windowed_labels), timestamps


    def create_training_data(self, data, labels, window_length=50, step_size=.25, sub_sample_length=25, clear_outliers=False, calc_features=False, keep_positions=False, padding_left_right=0, feature_thresholds = None):
        # slice into sliding windows
        data_windowed, labels_windowed, timestamps_windowed = self.slice_into_windows(data, labels, window_length=window_length, step_size=step_size)

        if calc_features:
            data_windowed = self.calculate_features(data_windowed, step_size, sub_sample_length, feature_thresholds, keep_positions=keep_positions)

        if padding_left_right > 0:
            for i in range(padding_left_right):
                data_windowed = np.insert(data_windowed, 0, 0, axis=1)
                data_windowed = np.insert(data_windowed, -1, 0, axis=1)

        result = [np.array(data_windowed), np.array(labels_windowed), timestamps_windowed, None]

        if clear_outliers:
            knn = KNN(contamination=.20, n_neighbors=10, method='mean', radius=1.0)
            knn.fit(data_windowed)

            # get outlier scores
            outlier_pred = knn.predict(data_windowed)
            cleared_data= []
            cleared_labels = []

            for i in range(len(outlier_pred)):
                # If it's not an outlier add it to the resulting array
                if outlier_pred[i] == 0:
                    cleared_data.append(data_windowed[i])
                    cleared_labels.append(labels_windowed[i])

            result = [np.array(cleared_data), np.array(cleared_labels), timestamps_windowed, knn]

        return result

    def evenly_oversample(self, X, y):
        X_result = []
        y_result = []

        unique, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        counts_dict = dict(zip(unique, counts))
        oversample_count = dict()

        for u in unique:
            oversample_count[u] = -max_count

        for i in range(0, X.shape[0]):
            label = y[i]
            multiply_factor = int(np.ceil(max_count / counts_dict[label]))
            for j in range(0, multiply_factor):
                if oversample_count[label] < 0:
                    X_result.append(X[i])
                    y_result.append(y[i])
                    oversample_count[label] += 1

        X_result = np.array(X_result)
        y_result = np.array(y_result)


        X_result, y_result = shuffle(X_result, y_result)

        return X_result, y_result

    def prepare_raw_data(self, files, columns, gopro_columns=[], gopro_sync_files=["accl", "gps", "gyro"], location_based_label_files=None, speed_threshold=1, fetch_from_apis=False, force_overwrite=True, min_cluster_size=3):
        print("Preparing raw data...")

        results = []
        resulting_labels = []

        for i in range(len(files)):
            file = files[i]
            file_name = 'data/' + file
            data = self.convert_and_read_fit_file(file_name)
            data = self.filter_data(data, speed_threshold=speed_threshold)
            data = self.split_hd_values(data)
            data = self.get_values_for(data, columns)
            data = np.array(data).T

            # Check if there is an mp4 file with the respective file_name
            # If so: Merge the values to the raw data
            if (os.path.isfile(file_name + '.mp4') and gopro_columns):
                print("Found mp4 file. Converting and syncing ...")
                gopro_data, gopro_csv_columns = self.convert_and_read_gopro_mp4_file(file_name)
                gopro_data = self.make_gopro_objects(gopro_data, gopro_csv_columns)
                gopro_data = self.get_values_for(gopro_data, gopro_columns, prepend_timestamps=False)
                gopro_data = np.array(gopro_data).T
                data = self.sync_data_with_gopro_data(data, gopro_data, distance_threshold=10)
                if not data:
                    print("Syncing with gopro data did not work. Shape: ", np.asarray(data).shape)
                    print("Maybe there were no matching data points")

            if not data:
                print("data object is empty, skipping...")
                continue

            results.append(data)

            # Check if there are location based labels
            if location_based_label_files is not None:
                # If the label assignment has been done before and force_overwrite is false: Load from the file
                label_file_name = file_name + '_labels'
                if os.path.isfile(label_file_name + '.npy') and not force_overwrite:
                    labels = np.load(label_file_name + '.npy')
                else:
                    location_based_label_file = location_based_label_files[i]
                    labels = self.label_data(data, location_based_label_file, min_cluster_size=min_cluster_size)
                    np.save(label_file_name, labels)
                resulting_labels.append(labels)

        print("Done")
        return results, resulting_labels


    def prepare_and_save_samples(self,
                             dataset_filename,
                             input_columns,
                             label_columns,
                             location_based_label_files = None,
                             window_lengths = [100, 200],
                             sub_sample_lengths = [25, 50],
                             step_size = .25,
                             auto_padd_left_right=False,
                             force_overwrite=False,
                             speed_threshold = .3,
                             min_cluster_size=3):

        dataset = pd.read_csv('data/' + dataset_filename + '.csv')
        dataset_recordings = [pd.DataFrame(y) for x, y in dataset.groupby('input_filename', as_index=False)]

        for window_length in window_lengths:
            for sub_sample_length in sub_sample_lengths:
                if sub_sample_length > window_length:
                    continue

                filename_raw = "data/%s_%s_%s_raw" % (dataset_filename, str(window_length),str(sub_sample_length))
                filename_features = "data/%s_%s_%s_features" % (dataset_filename, str(window_length),str(sub_sample_length))
                filename_labels = "data/%s_%s_%s_labels" % (dataset_filename, str(window_length),str(sub_sample_length))
                filename_timestamps = "data/%s_%s_%s_timestamps" % (dataset_filename, str(window_length),str(sub_sample_length))

                # Normalize, despite the first and the last three fields (Timestamp, Heading in rad, Latitude, Longitude)
                raw_scaler = StandardScaler()
                concatenated_data = dataset[input_columns].values[:, 1:-3]
                normalized_concatenated_data = raw_scaler.fit_transform(concatenated_data)

                # Calculate the upper and lower 30 percentiles, which are needed for the features
                lower_thresholds = np.percentile(normalized_concatenated_data, 30, axis=1)
                upper_thresholds = np.percentile(normalized_concatenated_data, 70, axis=1)

                raw_windowed, labels_windowed, features_windowed, timestamps_windowed = [], [], [], []

                for dataset_recording in dataset_recordings:

                    # Create an index mask on columns and map it to data_recording to get the data values
                    # Same for labels. Multiple labels can be stored in one file and picked later
                    training_data = dataset_recording[input_columns].values
                    labels = dataset_recording[label_columns].values

                    # Normalize, despite the first and last three fields using the pretrained StandardScaler (Timestamp, Heading in rad, Latitude, Longitude)
                    # TODO: If you remove the columns for heading, lat, lng, this won't work anymore
                    data_for_raw = np.hstack((training_data[:, :1], raw_scaler.transform(training_data[:, 1:-3]), training_data[:, -3:]))

                    padding_left_right = int((window_length%4)/2) if auto_padd_left_right else 0
                    raw_windowed_recording, labels_windowed_recording, timestamps_windowed_recording, _ = self.create_training_data(
                        data_for_raw,
                        labels,
                        window_length=window_length,
                        step_size=step_size,
                        sub_sample_length=sub_sample_length,
                        calc_features=False,
                        keep_positions=False,
                        padding_left_right=padding_left_right)

                    features_windowed_recording, _, _, _ = self.create_training_data(
                        training_data,
                        labels,
                        window_length=window_length,
                        step_size=step_size,
                        sub_sample_length=sub_sample_length,
                        feature_thresholds=[lower_thresholds, upper_thresholds],
                        calc_features=True,
                        keep_positions=True)

                    # Normalize, despite the last two fields (Latitude, Longitude)
                    feature_scaler = StandardScaler()
                    features_windowed_recording = np.hstack((feature_scaler.fit_transform(features_windowed_recording[:, :-2]), features_windowed_recording[:, -2:]))

                    raw_windowed.append(raw_windowed_recording)
                    labels_windowed.append(labels_windowed_recording)
                    features_windowed.append(features_windowed_recording)
                    timestamps_windowed.append(timestamps_windowed_recording)

                raw_windowed = np.concatenate(raw_windowed)
                labels_windowed = np.concatenate(labels_windowed)
                features_windowed = np.concatenate(features_windowed)
                timestamps_windowed = np.concatenate(timestamps_windowed)

                # Scale and save
                np.save(filename_raw, raw_windowed)
                np.save(filename_labels, labels_windowed)
                np.save(filename_features, features_windowed)
                np.save(filename_timestamps, timestamps_windowed)

                print("raw:", raw_windowed.shape)
                print("features:", features_windowed.shape)
                print("labels:", labels_windowed.shape, np.unique(labels_windowed, return_counts=True))
                print("--------------------------------\n")



