#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#CHANGE THIS to aboslute path of Caltech repo
os.chdir('/home/sabrar/256_ObjectCategories/')

#this is a global variable to set how many subset of classes we will work with 
#CHANGE THIS TO REQUIRED NUMBER (257) while running on GPU
working_classes=257


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

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


# In[ ]:


#globally setting up the square size that all images will be resized to
img_size=128


# In[ ]:


hm={}
path='./256_ObjectCategories'
folders=os.listdir(path)


for folder in folders:
    os.chdir(path+'/'+folder)
    image_paths=os.listdir()
    class_images=[]
    for i in image_paths:
        if i[-4:]=='.jpg':
            print("processing ",i)
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


# In[ ]:


#convert hm to X,y data mapping
#where each row of X is an image, y is it's class in int
import random
def make_Xy(hm, subset_classes):
    classes=list(hm.keys())
    random.shuffle(classes)
    classes=classes[0:subset_classes]
    X=[]
    y=[]
    for key in classes:
        images=hm[key]
        X.extend(images)
        y.extend([key]*len(images))
    X=np.array(X)
    y=pd.get_dummies(y)
    
    return X,y     


# In[ ]:


print("started")
X,y=make_Xy(hm,subset_classes=working_classes)
print(X.shape,y.shape)
#make a test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)
print("finished")


# In[ ]:


from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D,Flatten
from keras.applications.vgg16 import VGG16


# In[ ]:


from keras.callbacks import CSVLogger


# In[ ]:


def validation_plots(history, prefix):
    training_loss = history['loss']
    validation_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy=history['val_accuracy']
    epochs=range(1,len(accuracy)+1)
    
    plt.plot(epochs,training_loss,'bo',label='Training loss')
    plt.plot(epochs,validation_loss,'b',label='Test loss')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(prefix + '_1.png',dpi=500)
    plt.show()
    plt.clf()
    
    plt.plot(epochs,accuracy,'b',label='Training accuracy')
    plt.plot(epochs,val_accuracy,'r',label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(prefix + '_2.png',dpi=500)
    plt.show()
    plt.clf()


# In[ ]:


def TLmodel2(input_dim, output_classes):
    
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_dim)
    
    print(vgg_base.summary())
    
    model= Sequential()
    model.add(vgg_base)
    model.add(layers.Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(output_classes,activation='softmax'))

    print(len(model.trainable_weights))
    
    vgg_base.trainable=False
    
    print(len(model.trainable_weights))


    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model

model= TLmodel2((img_size,img_size,3), working_classes)
generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
csv_logger = CSVLogger('./Caltech256TL.csv', append=False, separator=',')
print("a")
fit=model.fit_generator(generator.flow(X_train, y_train.values, batch_size=32), 
                        epochs=10,verbose=1,validation_data=(X_test, y_test.values), callbacks=[csv_logger])
validation_plots(fit.history,"ANN")
print(model.evaluate(X_test,y_test))


# In[ ]:


def TLmodel3(input_dim, output_classes):
    
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_dim)
    
    print(vgg_base.summary())
    
    model= Sequential()
    model.add(vgg_base)
    model.add(layers.Flatten())
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(output_classes,activation='softmax'))

    print(len(model.trainable_weights))
    
    vgg_base.trainable=False
    
    print(len(model.trainable_weights))


    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model

model= TLmodel3((img_size,img_size,3), working_classes)
generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
csv_logger = CSVLogger('./Caltech256TLmultilayer.csv', append=False, separator=',')
print("a")
fit=model.fit_generator(generator.flow(X_train, y_train.values, batch_size=32), 
                        epochs=10,verbose=1,validation_data=(X_test, y_test.values), callbacks=[csv_logger])
validation_plots(fit.history,"TL_Multilayer")
print(model.evaluate(X_test,y_test))


# In[ ]:




