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

    def convert_and_read_fit_file(self, filename):
        print("Converting fit file", filename)

        converter = os.path.abspath("FitSDKRelease_21.16.00/java/FitCSVTool.jar")
        filepath = os.path.abspath(filename + ".fit")
        subprocess.run(["java", "-jar", converter,  filepath])
        data = pd.read_csv(filename + ".csv", low_memory=False)
        datav = data.query("Message == 'record'").values
        return datav

    def convert_and_read_gopro_mp4_file(self, filename, gopro_sync_files=["accl", "gps", "gyro"]):
        print("Converting gopro mp4 file", filename)

        # converter = os.path.abspath("gopro_converter.js")
        filepathMP4 = os.path.abspath(filename + ".MP4")
        filepathBin = os.path.abspath(filename + ".bin")
        filepathCsv = os.path.abspath(filename + "_gopro.csv")
        converter2bin = ["ffmpeg", "-y", "-i", filepathMP4, "-codec", "copy", "-map", "0:2", "-f", "rawvideo", filepathBin]
        converter2csv = ["gpmd2csv", "-i", filepathBin, "-o", filepathCsv]
        subprocess.run(converter2bin)
        subprocess.run(converter2csv)

        result = None

        for data_key in gopro_sync_files:
            filepathSubCsv = os.path.abspath(filename + "_gopro-" + data_key + '.csv')
            subValues = pd.read_csv(filepathSubCsv, low_memory=False)

            if('GpsAccuracy' in subValues.columns):
                subValues = subValues[subValues.GpsAccuracy < 300]

            if (result is None):
                result = subValues
            else:
                result = result.merge(subValues, left_on='Milliseconds', right_on='Milliseconds', how='left')

        # Interpolate Data
        for column in result.columns:
            result[column].interpolate(inplace=True, limit_direction='both')

        result.to_csv(filepathCsv)
        return result.values, result.columns

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

    def make_gopro_objects(self, gopro_values, gopro_csv_columns):
        result = []

        for row in gopro_values:
            current_object = {}
            for i in range(len(gopro_csv_columns)):
                column = gopro_csv_columns[i]
                current_object[column] = float(row[i])

            result.append(current_object)

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
                        result[int(timestamp) * 1000 + i*40] = new_row # TODO: changed *4 to *40, does this make sense???
            else:
                result[int(timestamp) * 1000] = row # TODO: Added *1000, does this make sense???

        return result

    def get_values_for(self, data, keys, prepend_timestamps=True):
        results = []
        data_values = list(data.values()) if isinstance(data,dict) else  data
        for key in keys:
            result = [row[key] if key in row else 0 for row in data_values]
            results.append(result)

        # Add the timestamp as first value
        if (prepend_timestamps):
            results = np.vstack((list(data.keys()), results))

        return results

    def calculate_features(self, samples, step_size, sub_sample_length, feature_thresholds, keep_positions=False):
        print("Calculating Features:", samples[0].shape[0], sub_sample_length)

        result = []

        lower_thresholds, upper_thresholds = feature_thresholds

        ACC_X = 0
        ACC_Y = 1
        ACC_Z = 2
        ALTITUDE = 3
        SPEED = 4
        HEART_RATE = 5
        # TODO: There can be all the newly added gopro data inbetween here
        HEADING = 6
        LAT = 7
        LNG = 8

        for i in tqdm(range(len(samples))):
            sample = samples[i]
            sample_slices, _, _ = self.slice_into_windows(sample, window_length=sub_sample_length, step_size=step_size)

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
            label = 0
            if i + window_length > data.shape[0]:
                window = data[-window_length:,  :]
                label = Counter(labels[-window_length:]).most_common()[0][0]
                #label = np.argmax(np.bincount(labels[-window_length:]))
                #label = round(np.mean(labels[-window_length:]))
            else:
                window = data[i:i + window_length, :]
                label = Counter(labels[i:i + window_length]).most_common()[0][0]
                #label = np.argmax(np.bincount(labels[i:i + window_length]))
                #label = round(np.mean(labels[i:i + window_length]))
            windows.append(window)
            windowed_labels.append(label)

        windows = np.array(windows)
        timestamps = windows[:, 0, 0] # seperate timestamps from data
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

    def label_data(self, data, labels_file_name, distance_threshold = 50, min_cluster_size=3):
        labels = []
        labels_csv = pd.read_csv('data/' + labels_file_name + '.csv')

        print("Labeling data by comparing locations ...")

        # Cleanup Labels
        labels_data = []
        last_coordinates = (0,0)
        for value_set in labels_csv.values:
            coordinates = (value_set[0], value_set[1])
            # Join the labels to one string
            new_value_set = np.hstack([value_set[:2], '-'.join(value_set[2:5]), value_set[-1]])

            # When changing labels, the app writes each change. Just take the latest label for one position in this case
            if coordinates == last_coordinates:
                labels_data[-1] = new_value_set
            else:
                labels_data.append(new_value_set)
                last_coordinates = coordinates

        labels_data = np.array(labels_data)
        unique_labels, counts = np.unique(labels_data[:, 2], return_counts=True)
        # Get rid of clusters smaller than min_cluster_size
        unique_labels = unique_labels[counts >= min_cluster_size]
        labels_data = list(filter(lambda dic: dic[2] in unique_labels, labels_data))
        print("RESULTING NUMBERS OF CLUSTERS: " + str(len(unique_labels)))

        last_label = 0
        last_location = None
        # Iterate through samples
        for i in tqdm(range(len(data))):
            sample = data[i]
            sample_location = (sample[-2], sample[-1])

            if last_location == sample_location:
                labels.append(last_label)
                break

            smallest_distance_label = (99999, 0)

            # Iterate through labels
            for label_data in labels_data:
                label_location = (label_data[0], label_data[1])

                # Calculate distance between label and sample
                distance = geodesic(sample_location, label_location).meters

                # Find the smallest distance
                if distance < distance_threshold and distance <= smallest_distance_label[0]:
                    smallest_distance_label = (distance, label_data[2])
                    last_label = smallest_distance_label[1]
                    last_location = label_location

            labels.append(smallest_distance_label[1])

        return labels

    def sync_data_with_gopro_data(self, data, gopro_data, distance_threshold=5): #sync_on_indices=[lat1, lng1, lat2, lng2]
        print("Syncing data and gopro data...")
        result = []

        last_latitude = 0
        last_longitude = 0
        last_item = []
        smallest_distance = 99999
        smallest_distance_timestamp=0
        last_latitude_gopro = 0
        last_longitude_gopro = 0

        for i in tqdm(range(len(data))):
            data_object = data[i]
            lat = data_object[-2]
            lng = data_object[-1]

            if (last_latitude == lat and last_longitude == lng):
                continue

            last_latitude = lat
            last_longitude = lng

            origin = (lat, lng)

            for gopro_data_object in gopro_data:
                gopro_lat = gopro_data_object[-2]
                gopro_lng = gopro_data_object[-1]

                if math.isnan(gopro_lat) or math.isnan(gopro_lng) or (last_latitude_gopro == gopro_lat and last_longitude_gopro == gopro_lng):
                    continue

                last_latitude_gopro = gopro_lat
                last_longitude_gopro = gopro_lng

                # Check if the item is in distance
                dest = (gopro_lat, gopro_lng)
                distance = geodesic(origin, dest).meters

                if (distance < smallest_distance):
                    smallest_distance = distance
                    smallest_distance_timestamp = gopro_data[0]

        # Loop through all data points
        for i in tqdm(range(len(data))):
            data_object = data[i]
            lat = data_object[-2]
            lng = data_object[-1]

            # If this is the same position as the one before, sync with the last item found
            if lat > 0 and lng > 0 and last_latitude == lat and last_longitude == lng and last_item is not []   :
                # TODO: This happens all the time
                result.append(np.hstack((data_object[:-3], last_item, data_object[-3:]))) #put heading, lat, lng in the end
                continue

            last_latitude = lat
            last_longitude = lng
            closest_item = None
            smallest_distance = distance_threshold
            origin = (lat, lng)

            # Loop through Gopro Telemetry data
            for gopro_data_object in gopro_data:
                gopro_lat = gopro_data_object[-2]
                gopro_lng = gopro_data_object[-1]

                if math.isnan(gopro_lat) or math.isnan(gopro_lng):
                    continue

                # Check if the item is in distance
                dest = (gopro_lat, gopro_lng)
                distance = geodesic(origin, dest).meters

                # If positions move closer together, pick the closer ones
                if distance < smallest_distance:
                    closest_item = gopro_data_object[:-2]
                    smallest_distance = distance
                elif closest_item is not None:
                    # If positions start moving further away, break TODO
                    break

            # if a close enough item was found merge with data and append lat/lng at the end
            if closest_item is not None:
                last_item = closest_item
                result.append(np.hstack((data_object[:-3], closest_item, data_object[-3:]))) #put heading, lat, lng in the end

        return result

    def prepare_and_save_samples(self,
                             files,
                             columns,
                             gopro_columns=[],
                             prefix='',
                             location_based_label_files = None,
                             window_lengths = [100, 200],
                             sub_sample_lengths = [25, 50],
                             step_size = .25,
                             auto_padd_left_right=False,
                             force_overwrite=False,
                             speed_threshold = .3,
                             min_cluster_size=3):

        filename_prepared_data = '%s_prepared_data' % prefix

        if os.path.isfile(filename_prepared_data + '.npy') and not force_overwrite:
            data, labels = np.load(filename_prepared_data + '.npy', allow_pickle=True)
        else:
            data, labels = self.prepare_raw_data(files, columns, gopro_columns=gopro_columns, speed_threshold=speed_threshold, location_based_label_files=location_based_label_files, min_cluster_size=min_cluster_size)
            np.save(filename_prepared_data, [data, labels])

        for window_length in window_lengths:
            for sub_sample_length in sub_sample_lengths:
                if sub_sample_length > window_length:
                    continue

                filename_raw = "data/%s_%s_%s_raw" % (prefix, str(window_length),str(sub_sample_length))
                filename_features = "data/%s_%s_%s_features" % (prefix, str(window_length),str(sub_sample_length))
                filename_labels = "data/%s_%s_%s_labels" % (prefix, str(window_length),str(sub_sample_length))
                filename_timestamps = "data/%s_%s_%s_timestamps" % (prefix, str(window_length),str(sub_sample_length))


                # Normalize, despite the first and the last three fields (Timestamp, Heading in rad, Latitude, Longitude)
                raw_scaler = StandardScaler()
                concatenated_data = np.concatenate(data)[:, 1:-3]
                normalized_concatenated_data = raw_scaler.fit_transform(concatenated_data)

                # Calculate the upper and lower 30 percentiles, which are needed for the features
                lower_thresholds = np.percentile(normalized_concatenated_data, 30, axis=1)
                upper_thresholds = np.percentile(normalized_concatenated_data, 70, axis=1)

                raw_windowed, labels_windowed, features_windowed, timestamps_windowed = [], [], [], []

                #for data_recording in data:
                for i in range(len(data)):

                    data_recording = np.asarray(data[i])
                    labels_recording = np.asarray(labels[i])

                    # Normalize, despite the first and last three fields using the pretrained StandardScaler (Timestamp, Heading in rad, Latitude, Longitude)
                    data_for_raw = np.hstack((data_recording[:, :1], raw_scaler.transform(data_recording[:, 1:-3]), data_recording[:, -3:]))

                    padding_left_right = int((window_length%4)/2) if auto_padd_left_right else 0
                    raw_windowed_recording, labels_windowed_recording, timestamps_windowed_recording, _ = self.create_training_data(
                        data_for_raw,
                        labels_recording,
                        window_length=window_length,
                        step_size=step_size,
                        sub_sample_length=sub_sample_length,
                        calc_features=False,
                        keep_positions=False,
                        padding_left_right=padding_left_right)

                    features_windowed_recording, _, _, _ = self.create_training_data(
                        data_recording,
                        labels_recording,
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



