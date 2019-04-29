import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
from tqdm import tqdm
import cv2, os
from helper import create_train_data

batch_size = 64
epochs = 10
num_classes = 2

TRAIN_DIR = "/Users/Pey/Downloads/train"
TEST_DIR  = "/Users/Pey/Downloads/test"
SIZE = 150


print("Loading Dataset...")

x_train, y_train, x_test, y_test = create_train_data(TRAIN_DIR)

print("x_train shape: {}\ny_train shape: {}\nx_test.shape: {}\ny_test.shape: {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

 
