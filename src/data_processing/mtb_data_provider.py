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

LATITUDE_KEY = 'position_lat'
LONGITUDE_KEY = 'position_long'


class MtbDataProvider:

    def convert_and_read_fit_file(self, filename):
        print("Converting fit file", filename)

        converter = os.path.abspath("FitSDKRelease_21.16.00/java/FitCSVTool.jar")
        filepath = os.path.abspath(filename + ".fit")
        subprocess.run(["java", "-jar", converter,  filepath])
        data = pd.read_csv(filename + ".csv")
        datav = data.query("Message == 'record'").values
        return datav      

    def filter_data(self, df):
        COLUMNS = ['distance', 'speed', 'heart_rate', 'altitude', 'SensorHeading', 'SensorAccelerationX_HD', 'SensorAccelerationY_HD', 'SensorAccelerationZ_HD', LATITUDE_KEY, LONGITUDE_KEY]
        SPEED_THRESHOLD = .5
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
            if current_object['speed'] >= SPEED_THRESHOLD:
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

    def get_values_for(self, data, key):
        return [row[key] for row in list(data.values()) if key in row]



