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

batch_size = 64
epochs = 10
num_classes = 2

TRAIN_DIR = "/Users/Pey/Downloads/train"
TEST_DIR  = "/Users/Pey/Downloads/test"
SIZE = 150

def create_train_data():
    images = []
    labels = []
    images1= []
    labels1= []
    for file in os.listdir(TRAIN_DIR):
        label = 0 if file.split('.')[0] == 'cat' else 1
        image = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR,file), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
        images.append(image)
        labels.append(label)
  
    #for file in os.listdir(TEST_DIR):
    #    label = 0 if file.split('.')[0] == 'cat' else 1
    #    image = cv2.resize(cv2.imread(os.path.join(TEST_DIR,file), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
    #    images1.append(image)
    #    labels1.append(label)

    
    return train_test_split((np.array(images)/255).reshape(-1, SIZE, SIZE, 1), to_categorical(labels), test_size=0.2)


print("Loading Dataset...")
x_train, y_train, x_test, y_test = create_train_data()

print("x_train shape: {}\ny_train shape: {}".format(x_train.shape, y_train.shape))





    
