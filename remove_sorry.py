from numpy  import *
import pandas as pd
import cv2
import simplejson
import numpy as np


# remove the one with the "no image found"
sorry_image = cv2.imread("image_vote_data/0/13.724841_100.473392.jpg")

data_number = 10

count = 0 
for i in range(0, data_number):
    directory = "training_list/"

    left_image_list = np.load(directory + "left_image_list_" + str(i) + ".npy")
    right_image_list = np.load(directory + "right_image_list_" + str(i) + ".npy")
    winner_list = np.load(directory + "winner_list_" + str(i) + ".npy")


    print(left_image_list.shape)
    print(right_image_list.shape)
    print(winner_list.shape)

    print(left_image_list)


    # let's remove the sorry images
    sorry_remove_list = []
    for j in range(len(left_image_list)):
        
        if (np.array_equal(left_image_list[j, :, :, :], sorry_image) or np.array_equal(right_image_list[j, :, :, :], sorry_image)):
            sorry_remove_list.append(j)



    cleaned_left_image_list = np.delete(left_image_list, sorry_remove_list, 0)
    cleaned_right_image_list = np.delete(right_image_list, sorry_remove_list, 0)
    cleaned_winner_list = np.delete(winner_list, sorry_remove_list, 0)

    print(left_image_list.shape)
    print(right_image_list.shape)
    print(winner_list.shape)

    print("----")    
    print(cleaned_left_image_list.shape)
    print(cleaned_right_image_list.shape)
    print(cleaned_winner_list.shape)

    # # write
    save("training_list/cleaned_left_image_list_" + str(i), cleaned_left_image_list)
    save("training_list/cleaned_right_image_list_" + str(i), cleaned_right_image_list)
    save("training_list/cleaned_winner_list_" + str(i), cleaned_winner_list)

