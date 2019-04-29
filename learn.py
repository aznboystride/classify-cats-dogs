import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
from tqdm import tqdm
import cv2, os
from helper import create_train_data

batch_size = 64
epochs = 10
num_classes = 2

TRAIN_DIR = "../train"
SIZE = 224


print("Loading Dataset...")

x_train, x_test, y_train, y_test = create_train_data(TRAIN_DIR, SIZE)

print("x_train shape: {}\ny_train shape: {}\nx_test.shape: {}\ny_test.shape: {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))


print("\nCreating convolutional neural network...")

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11,11), input_shape=(224, 224, 3), padding="valid", strides=(4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(4096))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(1000))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

print("Compiling model...")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])

print("Training model...")

model.fit(x_train, y_train, validation_split=.25, epochs=epochs)

