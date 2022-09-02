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

parser = argparse.ArgumentParser()
parser.add_argument('-m',"--model",help="input the name of pretrained model", default='ResNet50')
args = parser.parse_args()

def main():
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    # load the data
    train_files, train_targets = load_dataset('./dogImages/train')
    valid_files, valid_targets = load_dataset('./dogImages/valid')
    test_files, test_targets = load_dataset('./dogImages/test')
    dog_names = [item[20:-1] for item in sorted(glob("./dogimages/train/*/"))]

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train_tensors = paths_to_tensor(train_files)
    valid_tensors = paths_to_tensor(valid_files)
    test_tensors = paths_to_tensor(test_files)

    if args.model == "ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(133, activation='softmax')(x)
        MODEL = Model(inputs=model.input, outputs=predictions)
        MODEL.summary()
    elif args.model == "Xception":
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(133, activation='softmax')(x)
        MODEL = Model(inputs=model.input, outputs=predictions)
        MODEL.summary()
    elif args.model == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(133, activation='softmax')(x)
        MODEL = Model(inputs=model.input, outputs=predictions)
        MODEL.summary()
    elif args.model == "Resnet50V2":
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(include_top=False, weights='imagenet', input_shape =(224, 224, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(133, activation='softmax')(x)
        MODEL = Model(inputs=model.input, outputs=predictions)
        MODEL.summary()
    
    MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='saved_model/weights.best.'+ args.model+'.h5', 
                                verbose=1, save_best_only=True)

    MODEL.fit(train_tensors, train_targets, 
            validation_data=(valid_tensors, valid_targets),
            epochs=20, batch_size=32, callbacks=[checkpointer], verbose=1)
    
    acc = sum([np.argmax(MODEL.predict(np.expand_dims(tensor, axis=0))) == np.argmax(test_targets[i]) for i, tensor in enumerate(test_tensors)])/len(test_targets)
    print('accuracy: ', acc*100, '%')

if __name__=='__main__':
    main()