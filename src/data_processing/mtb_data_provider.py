# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
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

    def calculate_features(self, samples):
        result = []
        
        for sample in samples:
            feature_vector = []
            feature_vector.append(sample.mean())
            feature_vector.append(sample.std())
            feature_vector.append(sample.max())
            feature_vector.append(sample.min())
            feature_vector.append(sample.var())
            result.append(feature_vector)
        
        return np.array(result)


    def slice_into_windows(self, data, labels=None, window_length=25, step_size=1):
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


    def create_training_data(self, data, labels, window_length=50, step_size=.25, clear_outliers=True):
        # slice into sliding windows
        data_windowed, labels_windowed = self.slice_into_windows(data, labels, window_length=window_length, step_size=step_size)
        data_windowed = self.calculate_features(data_windowed)

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

    def prepare_raw_data(self, files, columns, speed_threshold=1, folder='data', transpose=False, fetch_from_apis=False):
        results = []
        concatenated = None

        for file in files:
            file_name = folder + '/' + file
            data = self.convert_and_read_fit_file(file_name)
            data = self.filter_data(data, speed_threshold=speed_threshold)
            data = self.split_hd_values(data)
            data = np.array(self.get_values_for(data, columns))

            if transpose:
                data = data.T

            results.append(data)

            if transpose:
                concatenated = np.concatenate((concatenated, data)) if concatenated is not None else data

        results.append(concatenated)

        print("Done")
        return results


