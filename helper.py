from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def create_train_data(TRAIN_DIR, SIZE=150):
    dirs = os.listdir('.data-set')
    if "x_train.npy" in dirs and "y_train.npy" in dirs and \
        "x_test.npy" in dirs and "y_test.npy" in dirs: 
        return np.load(".data-set/x_train.npy"), np.load(".data-set/y_train.npy"), np.load(".data-set/x_test.npy"), np.load(".data-set/y_test.npy")
 
    images = []
    labels = []
    
    for file in tqdm(os.listdir(TRAIN_DIR)):
        label = 0 if file.split('.')[0] == 'cat' else 1
        image = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR,file), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
        images.append(image)
        labels.append(label)
    images = (np.array(images)/255).reshape(-1, SIZE, SIZE, 1)
    labels = to_categorical(labels)
    data = train_test_split(images, labels, test_size=0.2)
    
    print("Saving data where needed")
 
    #if "x_train.npy" not in dirs: np.save(".data-set/x_train.npy", data[0])
    #if "y_train.npy" not in dirs: np.save(".data-set/y_train.npy", data[1])
    #if "x_test.npy" not in dirs: np.save(".data-set/x_test.npy", data[2])
    #if "y_test.npy" not in dirs: np.save(".data-set/y_test.npy", data[3])

    return data


  
