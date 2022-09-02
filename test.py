import tensorflow as tf
from sklearn.datasets import load_files 
from keras.utils import np_utils
import argparse
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt 
import cv2
from keras.preprocessing import image                  
from tqdm import tqdm
from tensorflow.keras.utils import load_img, img_to_array
from helper_func import path_to_tensor, paths_to_tensor, load_dataset

#### model ####
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from keras.models import Model

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--path", help="input the path of dog image")
parser.add_argument('-n', "--name", help="input the name of outcome")
parser.add_argument('-m',"--model",help="input the name of pretrained model", default='ResNet50')
args = parser.parse_args()

dog_names = [item[20:-1] for item in sorted(glob("./dogimages/train/*/"))]

if args.model == "ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(133, activation='softmax')(x)
        MODEL = Model(inputs=model.input, outputs=predictions)
        MODEL.load_weights('saved_model/weights.best.Resnet50.h5')

elif args.model == "Xception":
    from tensorflow.keras.applications.xception import Xception, preprocess_input
    model = Xception(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(133, activation='softmax')(x)
    MODEL = Model(inputs=model.input, outputs=predictions)
    MODEL.load_weights('saved_model/weights.best.Xception.h5')

elif args.model == "InceptionV3":
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    model = InceptionV3(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(133, activation='softmax')(x)
    MODEL = Model(inputs=model.input, outputs=predictions)
    MODEL.load_weights('saved_model/weights.best.InceptionV3.h5')

elif args.model == "ResNet50V2":
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
    model = ResNet50V2(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(133, activation='softmax')(x)
    MODEL = Model(inputs=model.input, outputs=predictions)
    MODEL.load_weights('saved_model/weights.best.ResNet50V2.h5')

img_path = args.path
img_tensor = path_to_tensor(img_path)
plt.title(dog_names[np.argmax(MODEL.predict(img_tensor))].split(".")[1])
img = cv2.imread(img_path)
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_rgb)
plt.savefig(args.name)