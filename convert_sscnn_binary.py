# convert winner to binary
import numpy as np
from numpy  import *


data_number = 10

for k in range(0, data_number):
    directory = "training_list/"    
    winner_list = np.load(directory + "sscnn_list_" + str(k) + ".npy")
    sscnn_list = []

    
    # output as sscnn 
    # left as [1, 0] / right as [0, 1]
    # convert to left as [1] / right as 0
    for index, winner in enumerate(winner_list):
        if np.array_equal(winner_list[index], [1, 0]):
            sscnn_element = 1
            sscnn_list.append(sscnn_element)

        elif np.array_equal(winner_list[index], [0, 1]):
            sscnn_element = 0
            sscnn_list.append(sscnn_element)
        else:
            print("oi")

    print(winner_list[0:5])
    print(sscnn_list[0:5])

    save("training_list/sscnn_binary_" + str(k), sscnn_list)

