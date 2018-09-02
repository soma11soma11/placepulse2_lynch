
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy  import *

london_data = pd.read_csv("degree_centrality/London.csv")
london_data = london_data.drop(['colors'], axis=1)
london_data = london_data.drop(['Unnamed: 0'], axis=1)
london_data = london_data.convert_objects(convert_numeric=True)
x = london_data.values 

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
london_data = pd.DataFrame(x_scaled)
london_data = london_data[0].tolist()
# print(london_data.shape)
# london_data.hist(bins=100)
# plt.show()



save("try_lynchian", london_data)








    


