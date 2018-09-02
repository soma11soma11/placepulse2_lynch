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
from keras.layers.core import Activation
from keras import optimizers



# define the input shape
image_input1 = Input(shape=(224, 224, 3), name="image_input1")
image_input2 = Input(shape=(224, 224, 3), name="image_input2")

# download the pretrained vgg model
shared_vgg = VGG19(weights='imagenet', include_top=False)

# freeze the VGG19 parts
# for layer in shared_vgg .layers:
#     layer.trainable = False

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

# construct the model
model = Model(inputs=[image_input1, image_input2], outputs=[ss_cnn_output])
model.summary()

# compile the model 
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(optimizer=sgd,
              loss={'sscnn_output': 'binary_crossentropy'},
              metrics=['accuracy'])


count = 0
# import data
sscnn_label = np.load("training_list/" + "sscnn_binary_" + str(count) + ".npy")
left_image = np.load("training_list/" + "final_left_image_list_" + str(count) + ".npy")
right_image = np.load("training_list/" + "final_right_image_list_" + str(count) + ".npy")

# train departs
csv_logger = CSVLogger('trained/history_' +str(count) +'.log')
history = model.fit(x = [left_image, right_image], y=[sscnn_label], epochs=5, validation_split=0.25, batch_size=32, callbacks=[csv_logger], shuffle=True) 



# save the model
open('trained/model_' +str(count) + '.json',"w").write(model.to_json())
model.save_weights('trained/weight_' + str(count) + '.hdf5')


# validation data
val_left_image = np.load("training_list/" + "final_left_image_list_" + str(10) + ".npy")[0:10]
val_right_image = np.load("training_list/" + "final_right_image_list_" + str(10) + ".npy")[0:10]


print("sscnn_output-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("sscnn_output").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

print("fustion_flatten-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("fustion_flatten").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

print("fusion_conv3-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("fusion_conv3").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

print("fusion_conv2-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("fusion_conv2").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

print("fusion_conv1-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("fusion_conv1").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

print("fusion_concatenated-----------------------------------------------")
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("fusion_concatenated").output)
intermediate_output = intermediate_layer_model.predict([val_left_image, val_right_image])
print(intermediate_output)
print(np.sum(intermediate_output))

