import math
import numpy as np
from matplotlib import pyplot as plt
import h5py
import random
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend as kb
from keras.models import model_from_json
from keras.initializers import RandomNormal
from keras.layers import *
from keras.applications.vgg16 import VGG16


kb.clear_session()
    
def initialize_weights(model):
    # vgg =  VGG16(weights='imagenet', include_top=False)
    
    json_file = open('C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/model_arc.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("C:/Users/Lenovo/Desktop/Crowd Computing using CSRNet/Weights/VGG_16.h5")
    
    # vgg = loaded_model
    
    # vgg_weights=[]                         
    for layer in vgg.layers:
        print(layer.name)
        # if('conv' in layer.name):
            # vgg_weights.append(layer.get_weights())   
    # print(len(vgg_weights))
    # offset=0
    # i=0
    # while(i<10):
    #     if('conv' in model.layers[i+offset].name):
    #         model.layers[i+offset].set_weights(vgg_weights[i])
    #         i=i+1
    #         #print('h')           
    #     else:
    #         offset=offset+1
    return (model)


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# Convolutional Neural Network model : first 10 layers of VGG16 + backend
def CSRNet():  

    # Variable Input Size
    rows, cols = None, None
    initializer = RandomNormal(stddev=0.01)
    model = Sequential() 
    
    # custom VGG:
    # frontend of 10 layers
    model.add(Conv2D(64, kernel_size = (3, 3),activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = initializer))
    model.add(Conv2D(64, kernel_size = (3, 3),activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(128,kernel_size = (3, 3), activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(Conv2D(128,kernel_size = (3, 3), activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(256,kernel_size = (3, 3), activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(Conv2D(256,kernel_size = (3, 3), activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(Conv2D(256,kernel_size = (3, 3), activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(MaxPooling2D(strides=2))            
    model.add(Conv2D(512, kernel_size = (3, 3),activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(Conv2D(512, kernel_size = (3, 3),activation = 'relu', padding='same', kernel_initializer = initializer))
    model.add(Conv2D(512, kernel_size = (3, 3),activation = 'relu', padding='same', kernel_initializer = initializer))
        
    # backend of 7 layers 
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = initializer, padding = 'same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = initializer, padding = 'same'))

    sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse'])
    
    model = initialize_weights(model)
    
    return model

model = CSRNet()
model.summary()
# train_gen = image_generator(img_paths,1)
# sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
# model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse'])
# model.fit_generator(train_gen,epochs=1,steps_per_epoch= 700 , verbose=1)
# save_mod(model,"weights/model_A_weights.h5","models/Model.json")