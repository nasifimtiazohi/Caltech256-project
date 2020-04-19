# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:32:32 2020

@author: tprotic
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D,Flatten

from keras.callbacks import CSVLogger

total_layers = 3
filters = 32
learning_rate = .001
epochs = 80
batch_size = 32
decay = learning_rate/epochs
optimizer = Adam(lr=learning_rate,decay=decay)

#CHANGE THIS to aboslute path of Caltech repo
os.chdir('C:/Users/proti/')
#this is a global variable to set how many subset of classes we will work with 
#CHANGE THIS TO REQUIRED NUMBER (257) while running on GPU
working_classes = 257

#globally setting up the square size that all images will be resized to
img_size=128

# this path is needed to read the images from each folder
path='./256_ObjectCategories'

#images are in different shapes. needs to square them.
def shrink_square(img, shrink_size, color_mode, fill_color):
    #all the images will be shrinked to shrink_size with any leftover filled with color_fill
    ## thumbnail resizes image while maintaining aspect ratio
    shrinked_img=img.thumbnail((shrink_size,shrink_size),Image.ANTIALIAS)
    
    #if some images were originally less than shrink_size, bring to shrink size by filling the leftover
    output=Image.new(color_mode,(shrink_size,shrink_size),fill_color)
    output.paste(img, (int((shrink_size - img.size[0]) / 2), int((shrink_size - img.size[1]) / 2)))
    
    return output


#turn images to np_array
def image_to_tensors(img):
    arr=np.array(img)
    return arr

def read_images():
    hm = {}
    folders=os.listdir(path)
    for folder in folders:
        os.chdir(path+'/'+folder)
        image_paths=os.listdir()
        class_images=[]
        for i in image_paths:
            if i[-4:]=='.jpg':
                #print("processing ",i)
                img=Image.open(i)
                shrinked_img=shrink_square(img, img_size, 'RGB', 0)
                img_arr=image_to_tensors(shrinked_img)
                img_arr=img_arr/255 #scaling pixel values to [0,1]
                class_images.append(img_arr)
                img.close()
                shrinked_img.close()
        class_name= folder.split('.')[1]
        hm[class_name]=class_images
        os.chdir('../..')
    return hm


#convert hm to X,y data mapping
#where each row of X is an image, y is it's class in int
def make_Xy_Dense(hm,subset_classes):
    classes=list(hm.keys())
    #random.shuffle(classes)
    classes=classes[0:subset_classes]
    X=[]
    y=[]
    for key in classes:
        images=hm[key]
        flatten_images=[]
        for i in images:
            flatten_images.append(i.flatten())
        X.extend(flatten_images)
        y.extend([key]*len(flatten_images))
    X=np.matrix(X)
    y=pd.get_dummies(y)
    return X,y

def createCNNModel(total_layers, filters, learning_rate, epochs, batch_size, decay, optimizer):
    model = Sequential()
    model.add(Conv2D(filters = (filters), kernel_size= (3,3), input_shape = (img_size,img_size,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    print('filter: '+ str(filters))

    for i in range(total_layers):
        fltr = filters * (2**(i+1))
        model.add(Conv2D(filters = fltr, kernel_size= (3,3), padding = 'same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size = (2,2)))
        print('filter: '+ str(filters * (2**(i+1))))

    print('\n')
    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 

    model.add(Dense(257,activation='softmax'))
    model.compile(optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    hm=read_images()
    X,y=make_Xy_Dense(hm,subset_classes=working_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(-1,img_size,img_size,3)
    X_test = X_test.reshape(-1,img_size,img_size,3)
    
    generator = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    model = createCNNModel(total_layers, filters, learning_rate, epochs, batch_size, decay, optimizer)
    csv_logger = CSVLogger('C:/Users/proti/MyJupyterNotebooks/Caltech256logDataAugmentation.csv', append=False, separator=',')
    from sklearn.metrics import classification_report,confusion_matrix
    model.fit_generator(generator.flow(X_train, y_train.values, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, 
                        epochs=80,
                        verbose=1,
                        validation_data=(X_test, y_test.values),
                        callbacks=[csv_logger])