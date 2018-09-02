from numpy  import *
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx


print("[49:56]")

city_list = pd.read_csv("city_list.csv", header=None)[0].tolist()

location_in_city_list = []
for city in city_list:
    location_in_city = []
    city_bbox = pd.read_csv("city_coords/" +str(city) + ".csv")
    north = city_bbox.loc[1, "max"]
    south = city_bbox.loc[1, "min"]
    east = city_bbox.loc[0, "max"]
    west = city_bbox.loc[0, "min"]
    bbox = [north, south, east, west]
    for count in range(0, 11):
        coords_left = np.load("../training_list/left_location" +str(count) + ".npy")
        coords_right = np.load("../training_list/right_location" +str(count) + ".npy")
        coords = np.concatenate([coords_left, coords_right])
        for point in coords:
            lat, long = point.decode('utf-8').split("_")
            lat = float(lat)
            long = float(long[0:-4])
            if (bbox[1] < lat < bbox[0]) and (bbox[3] < long < bbox[2]):
                location_in_city.append([lat, long])
            
    location_in_city_list.append(location_in_city)   

print(len(location_in_city_list))


for index, city in enumerate(city_list):

    print(city)

    coords_data = pd.read_csv("city_coords/" +str(city) + ".csv")
    north_bbox = coords_data.loc[1, "max"]
    south_bbox = coords_data.loc[1, "min"]
    east_bbox = coords_data.loc[0, "max"]
    west_bbox = coords_data.loc[0, "min"]
    G = ox.graph_from_bbox(north_bbox, south_bbox, east_bbox, west_bbox, network_type='walk')
   
    location_in_city = location_in_city_list[index]
    
    loc_list = [] 
    for location in location_in_city:
        nearest_node = ox.get_nearest_node(G, location)
        location_and_node = [location, nearest_node]
        loc_list.append(location_and_node)

    save("location_node/" + str(city), loc_list)





