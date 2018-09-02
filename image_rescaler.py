import cv2
import urllib
from numpy  import *
import numpy as np




count = 0
data_number

# rescale image to (224, 224)
for i in range(0,data_number):

    directory = "training_list/"

    new_left_image_list = []
    left_image_list = np.load(directory + "s_cleaned_left_image_list_" + str(i) + ".npy")
    for j in left_image_list:
        left_image_sized = cv2.resize(j, (224, 224)) 
        left_image_scaled = np.divide(left_image_sized, 255., dtype=np.float16)
        new_left_image_list.append(left_image_scaled)

        count += 1
        print(count)

    save(directory + "final_left_image_list_" + str(i), new_left_image_list)


    new_right_image_list = []
    right_image_list = np.load(directory + "s_cleaned_right_image_list_" + str(i) + ".npy")
    for j in right_image_list:
        right_image_sized = cv2.resize(j, (224, 224)) 
        right_image_scaled = np.divide(right_image_sized, 255., dtype=np.float16)
        new_right_image_list.append(right_image_scaled)

        count += 1
        print(count)

    save(directory + "final_right_image_list_" + str(i), new_right_image_list)



