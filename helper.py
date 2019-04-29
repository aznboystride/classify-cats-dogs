from tqdm import tqdm
import os
import cv2
import numpy as np

def create_train_data(TRAIN_DIR, SIZE=150):
    images = []
    labels = []
    images1= []
    labels1= []
    for file in tqdm(os.listdir(TRAIN_DIR)):
        label = 0 if file.split('.')[0] == 'cat' else 1
        image = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR,file), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE))
        images.append(image)
        labels.append(label)
  
    return train_test_split((np.array(images)/255).reshape(-1, SIZE, SIZE, 1), to_categorical(labels), test_size=0.2)

  
