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
lynch_input_left = Input(shape=(1,), name='lynch_input_left')
lynch_concatenated_left = Concatenate(name="lynch_concatenated_left")([rsscnn_fc1_2, lynch_input_left])
rsscnn_rank_left = Dense(1, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc1_3")(lynch_concatenated_left)


# second image
rsscnn_flattend2 = Flatten(name="rsscnn_flatten2")(vgg_output2)
rsscnn_fc2_1 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_1")(rsscnn_flattend2)
rsscnn_fc2_2 = Dense(4096, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_2")(rsscnn_fc2_1)
lynch_input_right = Input(shape=(1,), name='lynch_input_right')
lynch_concatenated_right = Concatenate(name="lynch_concatenated_right")([rsscnn_fc2_2, lynch_input_right])
rsscnn_rank_right = Dense(1, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal', name="rsscnn_fc2_3")(lynch_concatenated_right)

# # calculate the ranking difference between the image 1 and 2
rss_cnn_output = Subtract(name="rsscnn_output")([rsscnn_rank_right, rsscnn_rank_left])

# construct the model
model = Model(inputs=[image_input1, image_input2, lynch_input_left, lynch_input_right], outputs=[ss_cnn_output, rss_cnn_output])
model.summary()

# compile the model 
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(optimizer=sgd,
              loss={'sscnn_output': 'binary_crossentropy', 'rsscnn_output': "squared_hinge"},
              loss_weights={'sscnn_output': 1., 'rsscnn_output': 0.2},
              metrics=['accuracy'])


count = 1

# import data
sscnn_label = np.load("training_list/" + "sscnn_binary_" + str(count) + ".npy")[0:30]
rsscnn_label = np.load("training_list/" + "rsscnn_list_" + str(count) + ".npy")[0:30]
left_image = np.load("training_list/" + "final_left_image_list_" + str(count) + ".npy")[0:30]
right_image = np.load("training_list/" + "final_right_image_list_" + str(count) + ".npy")[0:30]
# lynch_left = np.load("LYNCHIAN/" + "try_lynchian.npy")[0:30]
# lynch_right = np.load("LYNCHIAN/" + "try_lynchian.npy")[0:30]
lynch_left = np.load("training_list/" + "degree_list_left" +str(count) + ".npy")[0:30]
lynch_right = np.load("training_list/" + "degree_list_right" +str(count) + ".npy")[0:30]

# train departs
csv_logger = CSVLogger('output/history_' +str(count) +'.log')
history = model.fit(x = [left_image, right_image, lynch_left, lynch_right], y=[sscnn_label, rsscnn_label], epochs=10, validation_split=0.25, batch_size=32, callbacks=[csv_logger], shuffle=True) 

# save the model
open('/output/model_' +str(count) + '.json',"w").write(model.to_json())
model.save_weights('/output/weight_' + str(count) + '.hdf5')





# floyd run --env keras --gpu \
#   --data 11soma/datasets/placepulse2/2:mounted_1 \
#   --data 11soma/datasets/placepulse2/3:mounted_2 \
#   "python3 rss_cnn.py"