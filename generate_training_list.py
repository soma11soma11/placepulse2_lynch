from numpy  import *
import pandas as pd
import cv2
import simplejson

count = 0 
data_number = 10
for i in range(0, data_number):

    winner_list = []
    left_image_list = []
    right_image_list = [] 

    directory = "image_vote_data/" +str(i) + "/"
    data = pd.read_csv(directory + "vote_with_image.csv")
    data = data.dropna()

    # winner
    winner = data["winner"].tolist()
    winner_list = winner_list + winner

    count += 1
    print(count)

    # image data left
    left_lat_list = data["left_lat"].tolist()
    left_long_list = data["left_long"].tolist()

    for left_lat, left_long in zip(left_lat_list, left_long_list):
        left_location = str(left_lat) + "_" + str(left_long) + ".jpg"
        left_image = cv2.imread(directory + left_location)

        left_image_list.append(left_image)

        count += 1
        print (count)

    # image data right
    right_lat_list = data["right_lat"].tolist()
    right_long_list = data["right_long"].tolist()

    for right_lat, right_long in zip(right_lat_list, right_long_list):
        right_location = str(right_lat) + "_" + str(right_long) + ".jpg"
        right_image = cv2.imread(directory + right_location)
        right_image_list.append(right_image)

        count += 1
        print (count)


    save("training_list/winner_list_" + str(i), winner_list)
    save("training_list/left_image_list_" + str(i), left_image_list)
    save("training_list/right_image_list_" + str(i), right_image_list)

