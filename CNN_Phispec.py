import os

USE_GPU = False

## Set up Tensorflow to use CPU or GPU
# Disable Tensorflow GPU computation (seems to break when loading inception)
if not USE_GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

from keras.optimizers import *

import numpy as np

##Tensorboard Utils
from keras.callbacks import TensorBoard
from time import time

batch_size = 64 # in each iteration, we consider 32 training examples at once
num_epochs = 20 # we iterate 20 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.25 # dropout in the FC layer with probability 0.5
hidden_size_1 = 512# the FC layer will have 512 neurons
hidden_size_2 = 32

#(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

input_data = np.load("nnetsetup/PhiTrainingData.npz")
X_train = input_data['X_train']
y_train = input_data['y_train']
X_test = input_data['X_test']
y_test = input_data['y_test']


num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


### Set up model ##
inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size_2, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model_final = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

#########################



####from keras import applications
####X_train_tmp = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3))
####for i in range(0, len(X_train)):
    ####for j in range(0, 3):
        ####X_train_tmp[i,:,:,j] = X_train[i, :, :, 0]
####X_train = X_train_tmp

####X_test_tmp = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3))
####for i in range(0, len(X_test)):
    ####for j in range(0, 3):
        ####X_test_tmp[i,:,:,j] = X_test[i, :, :, 0]
####X_test = X_test_tmp

####print(X_test.shape)

####model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape = (height, width, 3), classes=num_classes)

#####model = applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape = (height, width, 3), classes=num_classes)

####for layer in model.layers[:19]:
    ####layer.trainable = False

####x = model.output
####x = Flatten()(x)
####x = Dropout(drop_prob_1)(x)
####x = Dense(hidden_size_1, activation="relu")(x)
####x = Dropout(drop_prob_2)(x)
####x = Dense(hidden_size_2, activation="relu")(x)
####predictions = Dense(num_classes, activation="softmax")(x)

####model_final = Model(input = model.input, output = predictions)

model_final.summary()

########################
model_final.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer=Adamax(), # using the RMS optimiser
              metrics=['accuracy']) # reporting the accuracy

model_final.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.0) # ...holding out 15% of the data for validation


scores = model_final.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

print("\n%s: %.2f%%" % (model_final.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model_final.to_json()
with open("specmodel.nn", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("specmodel.h5")
print("Saved model to disk")

