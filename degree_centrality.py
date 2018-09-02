import networkx as nx
import osmnx as ox
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ox.config(use_cache=True, log_console=True)
# ox.__version__

city_list = pd.read_csv("city_list.csv", header=None)[0].tolist()

for index, city in enumerate(city_list):
    coords_data = pd.read_csv("city_coords/" +str(city) + ".csv")
    north_bbox = coords_data.loc[1, "max"]
    south_bbox = coords_data.loc[1, "min"]
    east_bbox = coords_data.loc[0, "max"]
    west_bbox = coords_data.loc[0, "min"]
    
    print(city)
    print(north_bbox, south_bbox, east_bbox, west_bbox)
    G = ox.graph_from_bbox(north_bbox, south_bbox, east_bbox, west_bbox, network_type='walk')


    approx = ox.basic_stats(G)["n"]/20
    node_centrality = nx.degree_centrality(G)

    df = pd.DataFrame(data=pd.Series(node_centrality).sort_values(), columns=['cc'])
    df['colors'] = ox.get_colors(n=len(df), cmap='inferno', start=0.2)
    df = df.reindex(G.nodes())
    nc = df['colors'].tolist()
    fig, ax = ox.plot_graph(G, bgcolor='k', node_size=4, node_color=nc, node_edgecolor='none', node_zorder=2,
                            edge_color="#46454b", edge_linewidth=1.5, edge_alpha=1, show=False, close=False)
    ax.set_title(city)
    fig.set_size_inches(4, 4)
    plt.savefig('degree_centrality/' +str(city)+ '.png')
    df.to_csv('degree_centrality/' +str(city)+ '.csv')




    

# floyd run \
#   --data 11soma/datasets/graph_lynch/1:mounted_1 \
#   "brew install spatialindex && python3 betweeness_centrality_graph.py"

