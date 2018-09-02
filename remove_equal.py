from numpy  import *
import pandas as pd
import cv2
import simplejson
import numpy as np

count = 0 
data_number = 10

for i in range(0, data_number):
    directory = "training_list/"

    left_image_list = np.load(directory + "cleaned_left_image_list_" + str(i) + ".npy")
    right_image_list = np.load(directory + "cleaned_right_image_list_" + str(i) + ".npy")
    winner_list = np.load(directory + "cleaned_winner_list_" + str(i) + ".npy")

    # let's remove the equal images
    equal_remove_list = []
    for j in range(len(winner_list)):
        if winner_list[j] == 'equal':
            equal_remove_list.append(j)

            print("oi_" + str(j))


    s_cleaned_left_image_list = np.delete(left_image_list, equal_remove_list, 0)
    s_cleaned_right_image_list = np.delete(right_image_list, equal_remove_list, 0)
    s_cleaned_winner_list = np.delete(winner_list, equal_remove_list, 0)


    print(left_image_list.shape)
    print(right_image_list.shape)
    print(winner_list.shape)

    print("----")    
    print(s_cleaned_left_image_list.shape)
    print(s_cleaned_right_image_list.shape)
    print(s_cleaned_winner_list.shape)

    # # write
    save("training_list/s_cleaned_left_image_list_" + str(i), s_cleaned_left_image_list)
    save("training_list/s_cleaned_right_image_list_" + str(i), s_cleaned_right_image_list)
    # save("training_list/s_cleaned_winner_list_" + str(i), s_cleaned_winner_list)

