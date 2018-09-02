from numpy  import *
import pandas as pd
import cv2
import simplejson
import numpy as np

########################################################################
data_number = 10
for data_num in range(0,data_number):
########################################################################
    print("removing the data outside of the city.....")

    city_list = pd.read_csv("LYNCHIAN/city_list.csv", header=None)[0].tolist()
    city_bbox_list = []
    for city in city_list:
        bbox = pd.read_csv("LYNCHIAN/city_coords/" +str(city) + ".csv")
        north = bbox.loc[1, "max"]
        south = bbox.loc[1, "min"]
        east = bbox.loc[0, "max"]
        west = bbox.loc[0, "min"]
        city_bbox = [north, south, east, west]
        city_bbox_list.append(city_bbox)


    directory = "image_vote_data/" +str(data_num) + "/"
    data = pd.read_csv(directory + "vote_with_image.csv")
    data = data.dropna()

    print(data.shape)


    # left image
    left_lat_list = data["left_lat"].tolist()
    left_long_list = data["left_long"].tolist()
    left_index_list = []
    for index, (left_lat, left_long) in enumerate(zip(left_lat_list, left_long_list)):
        for bbox in city_bbox_list:
            if (bbox[1] < left_lat < bbox[0]) and (bbox[3] < left_long < bbox[2]):
                left_index_list.append(index)

    # right image
    right_lat_list = data["right_lat"].tolist()
    right_long_list = data["right_long"].tolist()
    right_index_list = []
    for index, (right_lat, right_long) in enumerate(zip(right_lat_list, right_long_list)):
        for bbox in city_bbox_list:
            if (bbox[1] < right_lat < bbox[0]) and (bbox[3] < right_long < bbox[2]):
                right_index_list.append(index)

    left_right_index = list(set(left_index_list)&set(right_index_list))
    data_in_city = data.ix[left_right_index]

    print(data_in_city.shape)

    #########################################################################
    print("generating array from image.....")

    winner_list = []
    left_image_list = []
    right_image_list = []
    left_image_location_list = [] 
    right_image_location_list = [] 

    # winner
    winner = data_in_city["winner"].tolist()
    winner_list = winner_list + winner

    # image data_in_city_in_city left
    left_lat_list = data_in_city["left_lat"].tolist()
    left_long_list = data_in_city["left_long"].tolist()

    for left_lat, left_long in zip(left_lat_list, left_long_list):
        left_location = str(left_lat) + "_" + str(left_long) + ".jpg"
        left_image_location_list.append(left_location)
        left_image = cv2.imread(directory + left_location)
        left_image_list.append(left_image)
        
    # image data_in_city_in_city right
    right_lat_list = data_in_city["right_lat"].tolist()
    right_long_list = data_in_city["right_long"].tolist()

    for right_lat, right_long in zip(right_lat_list, right_long_list):
        right_location = str(right_lat) + "_" + str(right_long) + ".jpg"
        right_image_location_list.append(right_location)
        right_image = cv2.imread(directory + right_location)
        right_image_list.append(right_image)

    left_image_list = np.array(left_image_list)
    right_image_list = np.array(right_image_list) 
    winner_list = np.array(winner_list)
    left_image_location_list = np.array(left_image_location_list)
    right_image_location_list = np.array(right_image_location_list)

    print("----")   
    print(left_image_list.shape)
    print(right_image_list.shape)
    print(winner_list.shape)
    print(left_image_location_list.shape)
    print(right_image_location_list.shape)

    ###################################################################################################
    print("removing the one with the no image found.....")

    sorry_image = cv2.imread("image_vote_data/0/13.724841_100.473392.jpg")


    # let's remove the sorry images
    sorry_remove_list = []
    for j in range(len(left_image_list)):
        
        if (np.array_equal(left_image_list[j, :, :, :], sorry_image) or np.array_equal(right_image_list[j, :, :, :], sorry_image)):
            sorry_remove_list.append(j)

    cleaned_left_image_list = np.delete(left_image_list, sorry_remove_list, 0)
    cleaned_right_image_list = np.delete(right_image_list, sorry_remove_list, 0)
    cleaned_winner_list = np.delete(winner_list, sorry_remove_list, 0)
    cleaned_left_location_list = np.delete(left_image_location_list, sorry_remove_list, 0)
    cleaned_right_location_list = np.delete(right_image_location_list, sorry_remove_list, 0)

    print("----")   
    print(cleaned_left_image_list.shape)
    print(cleaned_right_image_list.shape)
    print(cleaned_winner_list.shape)
    print(cleaned_left_location_list.shape)
    print(cleaned_right_location_list.shape)

    ###################################################################################################
    print("removing equal.....")

    # let's remove the equal images
    equal_remove_list = []
    for j in range(len(cleaned_winner_list)):
        if cleaned_winner_list[j] == 'equal':
            equal_remove_list.append(j)


    s_cleaned_left_image_list = np.delete(cleaned_left_image_list, equal_remove_list, 0)
    s_cleaned_right_image_list = np.delete(cleaned_right_image_list, equal_remove_list, 0)
    s_cleaned_winner_list = np.delete(cleaned_winner_list, equal_remove_list, 0)
    s_cleaned_left_location_list = np.delete(cleaned_left_location_list, equal_remove_list, 0)
    s_cleaned_right_location_list = np.delete(cleaned_right_location_list, equal_remove_list, 0)


    print("----")    
    print(s_cleaned_left_image_list.shape)
    print(s_cleaned_right_image_list.shape)
    print(s_cleaned_winner_list.shape)
    print(s_cleaned_left_location_list.shape)
    print(s_cleaned_right_location_list.shape)

    save("training_list/left_location" + str(data_num), s_cleaned_left_location_list)
    save("training_list/right_location" + str(data_num), s_cleaned_right_location_list)

    ###################################################################################################
    print("convert winner to binary.....")

    sscnn_list = []
    rsscnn_list = []

    # output as sscnn 
    # left as [1, 0] / right as [0, 1]
    for n, i in enumerate(s_cleaned_winner_list):
        if i == "left":
            sscnn_element = [1, 0]
            sscnn_list.append(sscnn_element)

        elif i == "right":
            sscnn_element = [0, 1]
            sscnn_list.append(sscnn_element)

    save("training_list/sscnn_list_" + str(data_num), sscnn_list)

    # out put as rsscnn
    # left as [1] / right as [-1]
    for n, i in enumerate(s_cleaned_winner_list):
        if i == "left":
            rsscnn_element = 1
            rsscnn_list.append(rsscnn_element)

        elif i == "right":
            rsscnn_element = -1
            rsscnn_list.append(rsscnn_element)

    save("training_list/rsscnn_list_" + str(data_num), rsscnn_list)


    print("----")    
    print(np.array(sscnn_list).shape)
    print(np.array(rsscnn_list).shape)

    ###################################################################################################
    print("rescaling images.....")

    final_left_image_list = []
    for j in s_cleaned_left_image_list:
        left_image_sized = cv2.resize(j, (224, 224)) 
        left_image_scaled = np.divide(left_image_sized, 255., dtype=np.float16)
        final_left_image_list.append(left_image_scaled)

    save("training_list/final_left_image_list_" + str(data_num), final_left_image_list)

    final_right_image_list = []
    for j in s_cleaned_right_image_list:
        right_image_sized = cv2.resize(j, (224, 224)) 
        right_image_scaled = np.divide(right_image_sized, 255., dtype=np.float16)
        final_right_image_list.append(right_image_scaled)

    save("training_list/final_right_image_list_" + str(data_num), final_right_image_list)

    print("----")    
    print(np.array(final_left_image_list).shape)
    print(np.array(final_right_image_list).shape)

