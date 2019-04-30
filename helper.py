from tqdm import tqdm
import os
import cv2
import numpy as np
from random import randrange
#from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def create_train_data(TRAIN_DIR, SIZE=150):
    dirs = os.listdir('.data-set')
    if "x_train.npy" in dirs and "y_train.npy" in dirs and \
        "x_test.npy" in dirs and "y_test.npy" in dirs: 
        return np.load(".data-set/x_train.npy"), np.load(".data-set/y_train.npy"), np.load(".data-set/x_test.npy"), np.load(".data-set/y_test.npy")
 
    feature_labels = []

    for file in tqdm(os.listdir(TRAIN_DIR)):
        label = 0 if file.split('.')[0] == 'cat' else 1
        image = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR,file)), (SIZE, SIZE))
        feature_labels.append([image, label])
    
    data = train_test_split(feature_labels, SIZE, test_size=0.2) 
    #print("Saving data where needed")
 
    #if "x_train.npy" not in dirs: np.save(".data-set/x_train.npy", data[0])
    #if "y_train.npy" not in dirs: np.save(".data-set/y_train.npy", data[1])
    #if "x_test.npy" not in dirs: np.save(".data-set/x_test.npy", data[2])
    #if "y_test.npy" not in dirs: np.save(".data-set/y_test.npy", data[3])

    return data

def train_test_split(feature_labels, SIZE, test_size=0.2):
    data = []
    train_size = (1-test_size) * len(feature_labels)
    while len(data) < train_size:
        index = randrange(len(feature_labels))
        data.append(feature_labels.pop(index))
    x_train = [x[0] for x in data]
    y_train = [x[1] for x in data]
    x_test = [x[0] for x in feature_labels]
    y_test = [x[1] for x in feature_labels]
    print("Memory Before: {}".format(os.system('free -m')))
    x_train = (np.array(x_train)/255).reshape(-1, SIZE, SIZE, 3)
    x_test = (np.array(x_test)/255).reshape(-1, SIZE, SIZE, 3)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test
