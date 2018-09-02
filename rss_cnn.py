from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Concatenate, Conv2D, Dense, Flatten, Subtract
from keras import backend as K
import tensorflow as tf
import pandas as pd
import cv2
import urllib
from numpy  import *
import keras
import random
import pickle
from keras.callbacks import CSVLogger
from keras import regularizers
from keras import optimizers



# define the input shape
image_input1 = Input(shape=(224, 224, 3), name="image_input1")
image_input2 = Input(shape=(224, 224, 3), name="image_input2")

# download the pretrained vgg model
shared_vgg = VGG19(weights='imagenet', include_top=False)

# freeze the VGG19 parts
for layer in shared_vgg .layers:
    layer.trainable = False

# input each image to the vgg layer
vgg_output1 = shared_vgg(image_input1)
vgg_output2 = shared_vgg(image_input2)

# construct SS_CNN model
# and fusion (concatenate) them
fusion_concatenated = Concatenate(name="fusion_concatenated")([vgg_output1, vgg_output2])

# fusion network 
fusion_conv1 = Conv2D(1024, (3, 3), activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', name="fusion_conv1")(fusion_concatenated)
fusion_conv2 = Conv2D(1024, (3, 3), activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros', name="fusion_conv2")(fusion_conv1)
fusion_conv3 = Conv2D(1024, (3, 3), activation="tanh", kernel_initializer='random_uniform', bias_initializer='zeros', name="fusion_conv3")(fusion_conv2)
fusion_flattened = Flatten(name="fustion_flatten")(fusion_conv3)
ss_cnn_output = Dense(1, name="sscnn_output", activation="sigmoid", kernel_initializer='random_uniform', bias_initializer='zeros')(fusion_flattened)


# construct RSS_CNN model 
# first image
rsscnn_flattend1 = Flatten(name="rsscnn_flatten1")(vgg_output1)
rsscnn_fc1_1 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc1_1")(rsscnn_flattend1)
rsscnn_fc1_2 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc1_2")(rsscnn_fc1_1)
rsscnn_rank_left = Dense(1, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc1_3")(rsscnn_fc1_2)

# second image
rsscnn_flattend2 = Flatten(name="rsscnn_flatten2")(vgg_output2)
rsscnn_fc2_1 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_1")(rsscnn_flattend2)
rsscnn_fc2_2 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_2")(rsscnn_fc2_1)
rsscnn_rank_right = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_3")(rsscnn_fc2_2)

# # calculate the ranking difference between the image 1 and 2
rss_cnn_output = Subtract(name="rsscnn_output")([rsscnn_rank_right, rsscnn_rank_left])




# construct the model
model = Model(inputs=[image_input1, image_input2], outputs=[ss_cnn_output, rss_cnn_output])
model.summary()

# compile the model 
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(optimizer=sgd,
              loss={'sscnn_output': 'binary_crossentropy', 'rsscnn_output': "squared_hinge"},
              loss_weights={'sscnn_output': 1., 'rsscnn_output': 0.2},
              metrics=['accuracy'])



######### data concatenation left ############################################# 
print("concatenating left.........")
WHICH = "left"
directory = "training_list/final_" + str(WHICH) + "_image_list_"

data0_left = np.load(directory + str(0) + ".npy")[0:10]
data1_left = np.load(directory + str(1) + ".npy")[0:10]
data2_left = np.load(directory + str(2) + ".npy")[0:10]
data3_left = np.load(directory + str(3) + ".npy")[0:10]
data4_left = np.load(directory + str(4) + ".npy")[0:10]
data5_left = np.load(directory + str(5) + ".npy")[0:10]
data6_left = np.load(directory + str(6) + ".npy")[0:10]
data7_left = np.load(directory + str(7) + ".npy")[0:10]
data8_left = np.load(directory + str(8) + ".npy")[0:10]
data9_left = np.load(directory + str(9) + ".npy")[0:10]

concatenated_left = np.concatenate((data0_left, data1_left, data2_left, data3_left, data4_left, data5_left, data6_left, data6_left, data7_left, data8_left, data9_left), axis=0)


######### data concatenation right ############################################# 
print("concatenating right.........")
WHICH = "right"
directory = "training_list/final_" + str(WHICH) + "_image_list_"

data0_right = np.load(directory + str(0) + ".npy")[0:10]
data1_right = np.load(directory + str(1) + ".npy")[0:10]
data2_right = np.load(directory + str(2) + ".npy")[0:10]
data3_right = np.load(directory + str(3) + ".npy")[0:10]
data4_right = np.load(directory + str(4) + ".npy")[0:10]
data5_right = np.load(directory + str(5) + ".npy")[0:10]
data6_right = np.load(directory + str(6) + ".npy")[0:10]
data7_right = np.load(directory + str(7) + ".npy")[0:10]
data8_right = np.load(directory + str(8) + ".npy")[0:10]
data9_right = np.load(directory + str(9) + ".npy")[0:10]

concatenated_right = np.concatenate((data0_right, data1_right, data2_right, data3_right, data4_right, data5_right, data6_right, data6_right, data7_right, data8_right, data9_right), axis=0)


######### data concatenation rsscnn ############################################# 
print("concatenating rsscnn.........")
directory = "training_list/rsscnn_list_"

data0_rsscnn = np.load(directory + str(0) + ".npy")[0:10]
data1_rsscnn = np.load(directory + str(1) + ".npy")[0:10]
data2_rsscnn = np.load(directory + str(2) + ".npy")[0:10]
data3_rsscnn = np.load(directory + str(3) + ".npy")[0:10]
data4_rsscnn = np.load(directory + str(4) + ".npy")[0:10]
data5_rsscnn = np.load(directory + str(5) + ".npy")[0:10]
data6_rsscnn = np.load(directory + str(6) + ".npy")[0:10]
data7_rsscnn = np.load(directory + str(7) + ".npy")[0:10]
data8_rsscnn = np.load(directory + str(8) + ".npy")[0:10]
data9_rsscnn = np.load(directory + str(9) + ".npy")[0:10]

concatenated_rsscnn = np.concatenate((data0_rsscnn, data1_rsscnn, data2_rsscnn, data3_rsscnn, data4_rsscnn, data5_rsscnn, data6_rsscnn, data6_rsscnn, data7_rsscnn, data8_rsscnn, data9_rsscnn), axis=0)

######### data concatenation sscnn ############################################# 
print("concatenating sscnn..................")
directory = "training_list/sscnn_binary_" 

data0_sscnn = np.load(directory + str(0) + ".npy")[0:10]
data1_sscnn = np.load(directory + str(1) + ".npy")[0:10]
data2_sscnn = np.load(directory + str(2) + ".npy")[0:10]
data3_sscnn = np.load(directory + str(3) + ".npy")[0:10]
data4_sscnn = np.load(directory + str(4) + ".npy")[0:10]
data5_sscnn = np.load(directory + str(5) + ".npy")[0:10]
data6_sscnn = np.load(directory + str(6) + ".npy")[0:10]
data7_sscnn = np.load(directory + str(7) + ".npy")[0:10]
data8_sscnn = np.load(directory + str(8) + ".npy")[0:10]
data9_sscnn = np.load(directory + str(9) + ".npy")[0:10]

concatenated_sscnn = np.concatenate((data0_sscnn, data1_sscnn, data2_sscnn, data3_sscnn, data4_sscnn, data5_sscnn, data6_sscnn, data6_sscnn, data7_sscnn, data8_sscnn, data9_sscnn), axis=0)


print(concatenated_left.shape)
print(concatenated_right.shape)
print(concatenated_rsscnn.shape)
print(concatenated_sscnn.shape)


# train departs
csv_logger = CSVLogger('trained/history.log')
history = model.fit(x = [concatenated_left, concatenated_right], y=[concatenated_sscnn, concatenated_rsscnn], epochs=10, validation_split=0.25, batch_size=32, callbacks=[csv_logger], shuffle=True) 

# save the model
open('trained/model.json',"w").write(model.to_json())
model.save_weights('trained/weight.hdf5')

