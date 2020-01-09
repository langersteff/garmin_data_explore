
import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../src')
from matplotlib.pyplot import figure
import subprocess
import requests 
from xml.etree.ElementTree import fromstring, ElementTree
from geopy.distance import geodesic
import polyline

LATITUDE_KEY = 'position_lat'
LONGITUDE_KEY = 'position_long'
DISTANCE_META_THRESHOLD = 10 #in meters

class MtbDataTagger:

    def get_bounding_box_for_recording(self, latitudes, longitudes, mtb_data_provider, padding=0.):
        min_latitude = np.min(latitudes)
        max_latitude = np.max(latitudes)
        min_longitude = np.min(longitudes)
        max_longitude = np.max(longitudes)

        top_left = (max_latitude + padding, min_longitude - padding)
        bottom_right = (min_latitude - padding, max_longitude + padding)
        
        return (top_left, bottom_right)

    #https://overpass-api.de/api/map?bbox=11.7020,47.6756,11.9303,47.7750
    def fetch_area_from_openstreetmap(self, top_left, bottom_right):
        bbox= 'bbox=' + str(top_left[1]) + ',' + str(bottom_right[0]) + ',' + str(bottom_right[1]) + ',' + str(top_left[0])
        url = 'https://overpass-api.de/api/map?' + bbox
        print("fetching from: ", url)
        response = requests.get(url)
        return response

    #https://www.trailforks.com/api/1/maptrails?output=encoded&filter=bbox%3A%3A47.41852456703782%2C12.356023974716663%2C47.41852456703782%2C12.356023974716663&api_key=docs
    def fetch_area_from_trailforks(self, top_left, bottom_right):
        bbox= 'bbox::' + str(top_left[0]) + ',' + str(top_left[1]) + ',' + str(bottom_right[0]) + ',' + str(bottom_right[1])
        url= 'https://www.trailforks.com/api/1/trails?scope=track&filter=' + bbox + '&app_secret=374340a0656feca4&app_id=45'
        print("fetching from: ", url)
        response = requests.get(url)
        return response

    def create_trailforks_meta(self, top_left, bottom_right, response = None):
        if response is None:
            response = fetch_area_from_trailforks(top_left, bottom_right)
        datas = response.json()['data']

        for data in datas:
            encoded_path = data['track']['encodedPath']
            decoded_path = polyline.decode(encoded_path)
            del data['track']
            data['positions'] = decoded_path
        
        return datas

    def create_openstreetmap_meta(self, top_left, bottom_right, response = None):
        if response is None:
            response = fetch_area_from_openstreetmap(top_left, bottom_right)
        tree = ElementTree(fromstring(response.content))
        root = tree.getroot()
        position_nodes = {}
        mapped_nodes = {}

        for child in root:
            if (child.tag == 'node'):
                position_nodes[child.attrib['id']] = {'lat': child.attrib['lat'], 'lon': child.attrib['lon']}

        for child in root:
            if (child.tag == 'way'):
                metas = {}

                for sub_child in child:
                    if (sub_child.tag == 'tag'):
                        metas[sub_child.attrib['k']] = sub_child.attrib['v']

                for sub_child in child:
                    if (sub_child.tag == 'nd'):
                        node = position_nodes[sub_child.attrib['ref']]
                        mapped_nodes[sub_child.attrib['ref']] = {**node, **metas}
                        
        return mapped_nodes

    def find_meta_data_for_recording(self, latitudes, longitudes, openstreetmap_meta, trailforks_meta, mtb_data_provider):
        #latitudes = mtb_data_provider.get_values_for(data, LATITUDE_KEY)[1::25]
        #longitudes = mtb_data_provider.get_values_for(data, LONGITUDE_KEY)[1::25]
        closest_items = []
        
        for i in range(len(latitudes)):
            lat = latitudes[i]
            lon = longitudes[i]
            closest_item = {}
            closest_item_trailforks = {}
            smallest_distance = DISTANCE_META_THRESHOLD
            smallest_distance_trailforks = DISTANCE_META_THRESHOLD
            origin = (lat, lon)

            #TODO: Rename variables
            # Loop through OSM data
            for _, position_meta in openstreetmap_meta.items():
                
                    if 'mtb:scale' not in position_meta.keys() and 'mtb:type' not in position_meta.keys():
                        continue
                    
                    # Check if the item is in distance
                    dest = (position_meta['lat'], position_meta['lon'])
                    distance = geodesic(origin, dest).meters
                    
                    # Set as the closest OSM item
                    if distance < smallest_distance:
                        closest_item = position_meta
                        smallest_distance = distance
                        
            # Loop through Trailforks data
            for position_meta_trailforks in trailforks_meta:
                # Loop through positions in trailforks data
                for latitude, longitude in position_meta_trailforks['positions']:
                    
                    # Check if the item is in distance
                    dest = (latitude, longitude)
                    distance = geodesic(origin, dest).meters

                    # Set as the closest trailforks item
                    if distance < smallest_distance_trailforks:
                        closest_item_trailforks = position_meta_trailforks
                        smallest_distance_trailforks = distance
                        
            if closest_item or closest_item_trailforks:
                closest_items.append({**closest_item, **closest_item_trailforks})
            elif len(closest_items) > 0:
                closest_items.append(closest_items[-1])
            else:
                closest_items.append({})
                
        for closest_item in closest_items:
            if 'encodedPath' in closest_item:
                del closest_item['encodedPath']
            if 'positions' in closest_item:
                del closest_item['positions']
            if 'encodedLevels' in closest_item:
                del closest_item['encodedLevels']
            
        return closest_items

