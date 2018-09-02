from keras.models import model_from_json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import Model
from numpy  import *


model_type = "rsscnn"
print(model_type)

# load json and create model
json_file = open('trained/final/' + model_type + '_model/model_9.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("trained/final/" + model_type + "_model/weight_9.hdf5")


left_original_image_list = np.load("training_list/" + "final_left_image_list_" + str(10) + ".npy")
right_original_image_list = np.load("training_list/" + "final_right_image_list_" + str(10) + ".npy")

for left_image in left_original_image_list:
    randin


# lynch_left = np.load("training_list/" + "degree_list_left" +str(10) + ".npy")
# lynch_right = np.load("training_list/" + "degree_list_right" +str(10) + ".npy")
 
# predict
results = loaded_model.predict([left_image, right_image])
print(results)

