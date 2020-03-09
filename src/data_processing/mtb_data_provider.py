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

LATITUDE_KEY = 'position_lat'
LONGITUDE_KEY = 'position_long'


class MtbDataProvider:

    def convert_and_read_fit_file(self, filename):
        print("Converting fit file", filename)

        converter = os.path.abspath("FitSDKRelease_21.16.00/java/FitCSVTool.jar")
        filepath = os.path.abspath(filename + ".fit")
        subprocess.run(["java", "-jar", converter,  filepath])
        data = pd.read_csv(filename + ".csv", low_memory=False)
        datav = data.query("Message == 'record'").values
        return datav

    def filter_data(self, df, speed_threshold=0):
        COLUMNS = ['distance', 'speed', 'heart_rate', 'altitude', 'SensorHeading', 'SensorAccelerationX_HD', 'SensorAccelerationY_HD', 'SensorAccelerationZ_HD', LATITUDE_KEY, LONGITUDE_KEY]
        result = {}

        for row in df:
            current_object = {}
            current_objects = []
            current_timestamp = 0
            for i in range(len(row)):
                column = row[i]

                if column == 'timestamp':
                    current_timestamp = row[i+1]
                elif column in COLUMNS:
                    if column.endswith("_HD"):
                        current_object[column] = row[i+1]
                    # lat/long is written in semicircles
                    elif column in [LATITUDE_KEY, LONGITUDE_KEY]:
                        current_object[column] = float(row[i+1]) * 180.0 / 2**31
                    else:
                        current_object[column] = float(row[i+1])

            # SPEED THRESHOLD
            if current_object['speed'] >= speed_threshold:
                result[current_timestamp] = current_object

        return result

    def split_hd_values(self, data):
        result = {}
        for timestamp, row in data.items():

            if 'SensorAccelerationX_HD' in row:
                if (type(row['SensorAccelerationX_HD']) is str):
                    hd_values_x = row['SensorAccelerationX_HD'].split('|')
                    hd_values_y = row['SensorAccelerationY_HD'].split('|')
                    hd_values_z = row['SensorAccelerationZ_HD'].split('|')

                    for i in range(len(hd_values_x)):
                        new_row = row.copy()
                        new_row['SensorAccelerationX_HD'] = float(hd_values_x[i])
                        new_row['SensorAccelerationY_HD'] = float(hd_values_y[i])
                        new_row['SensorAccelerationZ_HD'] = float(hd_values_z[i])
                        result[int(timestamp) * 1000 + i*4] = new_row
            else:
                result[timestamp] = row

        return result

    def get_values_for(self, data, keys):
        results = []
        for key in keys:
            result = [row[key] if key in row else 0 for row in list(data.values())]
            results.append(result)
        return results

    def calculate_features(self, samples, step_size, sub_sample_length, keep_positions=False):
        print("Calculating Features")

        result = []

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
            sample_slices, _ = self.slice_into_windows(sample, window_length=sub_sample_length, step_size=step_size)

            feature_vector = []
            # feature_vector.append(sample[:, ACC_X].mean())
            # feature_vector.append(sample[:, ACC_Y].mean())
            # feature_vector.append(sample[:, ACC_Z].mean())
            # feature_vector.append(sample[:, ACC_X].max())
            # feature_vector.append(sample[:, ACC_Y].max())
            # feature_vector.append(sample[:, ACC_Z].max())

            feature_vector.append(sample[:, ACC_X:ACC_Z].mean())
            feature_vector.append(sample[:, ACC_X:ACC_Z].max())
            feature_vector.append(sample[:, ACC_X:ACC_Z].min())
            #feature_vector.append(sample[:, ACC_X:ACC_Z].std())

            # max altitude change, in sub samples
            feature_vector.append(np.max([sample_slices[:, :, ALTITUDE].max(axis=1) - sample_slices[:, :, ALTITUDE].min(axis=1)]))

            # max speed change, in sub samples
            feature_vector.append(np.max([sample_slices[:, :, SPEED].max(axis=1) - sample_slices[:, :, SPEED].min(axis=1)]))

            # mean speed on the trail
            feature_vector.append(sample[:, SPEED].mean()) # speed mean

            # max heading change per distance
            feature_vector.append(self.calc_max_heading_change_per_distance(sample_slices, LAT, LNG, HEADING))

            # max heading change, in sub samples
            #feature_vector.append(np.max([sample_slices[:, :, HEADING].max(axis=1) - sample_slices[:, :, HEADING].min(axis=1)]))

            if keep_positions:
                feature_vector.append(sample[:, LAT].mean())
                feature_vector.append(sample[:, LNG].mean())

            result.append(feature_vector)

        # Normalize all data despite the LAT LNG corrdinates
        result = np.array(result)
        normalized_features = normalize(result[:, :-2], axis=0)
        result = np.hstack((normalized_features, result[:, -2:]))

        return result


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
                # TODO: is [0] the right way to go?
                sample1_index = np.where(heading_slice == max_delta_combination[0])[0][0]
                sample2_index = np.where(heading_slice == max_delta_combination[1])[0][0]

                # Find the corresponding element within the full samples (not only heading)
                sample1 = sample_slice[sample1_index]
                sample2 = sample_slice[sample2_index]

                # Calculate the distance between those two elements
                location1 = (sample1[LAT], sample1[LNG])
                location2 = (sample2[LAT], sample2[LNG])
                distance = geodesic(location1, location2).meters

                # Calculate the ratio (how tight a corner is)
                max_heading_distance_ratio = np.max([max_delta/distance, max_heading_distance_ratio])

        return max_heading_distance_ratio

    # Hadings are given in rad. This calculates the angle between two rad values properly
    def calc_heading_delta(self, heading):
        heading_delta = heading[1] - heading[0]

        while (heading_delta < -math.pi):
            heading_delta += 2*math.pi
        while (heading_delta > math.pi):
            heading_delta -= 2*math.pi

        return np.abs(heading_delta)

    def slice_into_windows(self, data, labels=None, window_length=50, step_size=1):
        windows = []
        windowed_labels = []

        if labels is None:
            labels = np.zeros(len(data))

        stride = int(window_length // (1 // step_size))
        for i in range(0, data.shape[0], stride):
            label = labels[i]
            if i + window_length > data.shape[0]:
                window = data[-window_length:,  :]
            else:
                window = data[i:i + window_length, :]
            windows.append(window)
            windowed_labels.append(label)

        return np.array(windows), np.array(windowed_labels)


    def create_training_data(self, data, labels, window_length=50, step_size=.25, sub_sample_length=25, clear_outliers=False, calc_features=False, keep_positions=False, padding_left_right=0):
        # slice into sliding windows
        data_windowed, labels_windowed = self.slice_into_windows(data, labels, window_length=window_length, step_size=step_size)

        if calc_features:
            data_windowed = self.calculate_features(data_windowed, step_size, sub_sample_length, keep_positions=keep_positions)
        elif padding_left_right > 0:
            for i in range(padding_left_right):
                data_windowed = np.insert(data_windowed, 0, 0, axis=1)
                data_windowed = np.insert(data_windowed, -1, 0, axis=1)

        if clear_outliers == False:
            return np.array(data_windowed), np.array(labels_windowed), None

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


        return [np.array(cleared_data), np.array(cleared_labels), knn]

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

    def prepare_raw_data(self, files, columns, location_based_label_files=None, speed_threshold=1, folder='data', transpose=False, fetch_from_apis=False):
        results = []
        resulting_labels = []
        concatenated = None
        concatenated_labels = None

        for i in range(len(files)):
            file = files[i]
            file_name = folder + '/' + file
            data = self.convert_and_read_fit_file(file_name)
            data = self.filter_data(data, speed_threshold=speed_threshold)
            data = self.split_hd_values(data)
            data = np.array(self.get_values_for(data, columns))

            if transpose:
                data = data.T

            results.append(data)

            # Check if there are location based labels
            if location_based_label_files is not None:
                location_based_label_file = location_based_label_files[i]
                labels = self.label_data(data, location_based_label_file)
                resulting_labels.append(labels)

            if transpose:
                concatenated = np.concatenate((concatenated, data)) if concatenated is not None else data
                if location_based_label_files is not None:
                    concatenated_labels = np.concatenate((concatenated_labels, labels)) if concatenated_labels is not None else data

        results.append(concatenated)
        resulting_labels.append(concatenated_labels)

        print("Done")
        return results, resulting_labels

    def label_data(self, data, labels_file_name, distance_threshold = 6, accuracy_threshold=20, n_nearest_labels=5):
        labels = []
        labels_csv = pd.read_csv(labels_file_name)

        # Iterate through samples
        for i in len(range(data)):
            sample = data[i]
            sample_location = (sample[-2], sample[-1])

            # Iterate through labels
            for index, label_row in labels.iterrows():
                label_location = (label_row['latitude'], label_row['longitude'])
                accuracy = labels['accuracy']

                # Discard samples with low accuracy
                if accuracy > accuracy_threshold:
                    continue

                # Calculate distance between label and sample
                distance = geodesic(sample_location, label_location).meters

                smallest_distances = dict()
                # If the distance is small enough, remember it
                if distance <= distance_threshold:
                    smallest_distances[str(distance)] = label_row['label']

                    # If there are n labels with low enough distance
                    if len(smallest_distances.keys()) == n_nearest_labels:
                        # Find the smallest distance
                        float_smallest_distances = np.asarray(smallest_distances.keys, dtype=np.float32)
                        smallest_distance_label = smallest_distances[str(np.min(float_smallest_distances))]
                        # Set the respective label
                        labels.append(smallest_distance_label)
                        smallest_distances = dict()
                        # Jump to next sample
                        break
                        # TODO, Do not iterate the whole file each time
                        # i += 1
                        # continue
        return labels

    def prepare_and_save_samples(self,
                             files,
                             columns,
                             prefix='',
                             location_based_label_files = None,
                             window_lengths = [100, 200],
                             sub_sample_lengths = [25, 50],
                             step_size = .25,
                             force_overwrite=True,
                             clear_outliers = False,
                             calc_features = True,
                             keep_positions = True,
                             padding_left_right = 1,
                             speed_threshold = .3):

        data_all, labels_all = self.prepare_raw_data(files, columns, speed_threshold=speed_threshold, location_based_label_files=location_based_label_files, transpose=True)
        data_all = data_all[-1]
        labels_all = labels_all[-1]

        for window_length in window_lengths:
            for sub_sample_length in sub_sample_lengths:
                if sub_sample_length >= window_length:
                    continue

                filename_raw = "data/%s_%s_%s_raw" % (prefix, str(window_length),str(sub_sample_length))
                filename_features = "data/%s_%s_%s_features" % (prefix, str(window_length),str(sub_sample_length))
                filename_labels = "data/%s_%s_%s_labels" % (prefix, str(window_length),str(sub_sample_length))

                if not os.path.isfile(filename_raw) or  not os.path.isfile(filename_labels) or force_overwrite:
                    padding_left_right = int((window_length%4)/2)
                    raw_windowed, labels_windowed, _ = self.create_training_data(data_all, labels_all, window_length=window_length, step_size=step_size, sub_sample_length=sub_sample_length, clear_outliers=clear_outliers, calc_features=False, keep_positions=keep_positions, padding_left_right=padding_left_right)
                    np.save(filename_raw, raw_windowed)
                    np.save(filename_labels, labels_windowed)

                if not os.path.isfile(filename_features) or force_overwrite:
                    features_windowed, _, _ = self.create_training_data(data_all, labels_all, window_length=window_length, step_size=step_size, sub_sample_length=sub_sample_length, clear_outliers=clear_outliers, calc_features=True, keep_positions=keep_positions)
                    np.save(filename_features, features_windowed)


                print("raw:", raw_windowed.shape)
                print("features:", features_windowed.shape)
                print("labels:", labels_windowed.shape)
                print("--------------------------------\n")



