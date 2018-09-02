import pandas as pd
import numpy as np
from numpy  import *

city_list = pd.read_csv("city_list.csv", header=None)[0].tolist()

location_degree_list=[]

for city in city_list:
    print(city)
    # location and node
    location_node = np.load("location_node/" + str(city) + ".npy")

    # node and degree
    node_degree = pd.read_csv("degree_centrality/" + str(city) + ".csv")
    node_degree = node_degree.drop(['colors'], axis=1)
    node_degree = node_degree.rename(columns={'Unnamed: 0': 'node', 'cc': 'degree'})


    node_list = location_node[:,1]
    degree_list = []
    for node in node_list:
        degree = node_degree[node_degree["node"] == node]["degree"].values
        degree_list.append(degree)

    location_list = np.delete(location_node, 1, 1) 
    location_list = location_list.tolist()

    degree_location = pd.DataFrame(
        {'degree': degree_list,
        'location': location_list,
        })

    location_degree_list.append(degree_location)

location_degree_list = pd.concat(location_degree_list, axis=0)

location_degree_list.to_csv("location_degree_list.csv")