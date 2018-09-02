# convert winner to binary
import numpy as np
from numpy  import *


data_number = 10

for k in range(0, data_number):
    directory = "training_list/"    
    winner_list = np.load(directory + "s_cleaned_winner_list_" + str(k) + ".npy")

    sscnn_list = []
    rsscnn_list = []
    
    # output as sscnn 
    # left as [1, 0] / right as [0, 1]
    for n, i in enumerate(winner_list):
        if i == "left":
            sscnn_element = [1, 0]
            sscnn_list.append(sscnn_element)

        elif i == "right":
            sscnn_element = [0, 1]
            sscnn_list.append(sscnn_element)
        else:
            print("oi")

    save("training_list/sscnn_list_" + str(k), sscnn_list)

    # out put as rsscnn
    # left as [1] / right as [-1]
    for n, i in enumerate(winner_list):
        if i == "left":
            rsscnn_element = 1
            rsscnn_list.append(rsscnn_element)

        elif i == "right":
            rsscnn_element = -1
            rsscnn_list.append(rsscnn_element)
        else:
            print("oi")

    save("training_list/rsscnn_list_" + str(k), rsscnn_list)