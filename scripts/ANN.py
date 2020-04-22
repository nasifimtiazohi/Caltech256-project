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


X,y=make_Xy(hm,subset_classes=working_classes)
#make a test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)


# In[ ]:


from keras.models import Sequential
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


# In[ ]:


from keras.callbacks import CSVLogger


# In[ ]:


def DenseNetwork(X_train,y_train,input_dim,output_classes,
                 hidden_layer=2,units=[2048,2048], kernel_initializer='RandomNormal', 
                 hidden_layer_activation='relu',output_layer_activation='softmax', 
                 loss='categorical_crossentropy',metrics=['accuracy'],learning_rate=.1,
                 dropout=.5,batch_size=32,epochs=80):
    
    csv_logger = CSVLogger('./Caltech256ANN.csv', append=False, separator=',')
    
    #flatten the images first for dense network
    X_train=X_train.reshape(X_train.shape[0],-1)
    
    model= Sequential()
    
    for i in range(0,hidden_layer):
        model.add(Dense(units=units[0],kernel_initializer=kernel_initializer,input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Activation(hidden_layer_activation))
        model.add(Dropout(dropout))


    model.add(Dense(output_classes,kernel_initializer=kernel_initializer,activation=output_layer_activation))
    model.compile(Adam(lr=learning_rate),loss=loss,metrics=metrics)

    fit=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                  validation_data=(X_test.reshape(X_test.shape[0],-1),y_test), callbacks=[csv_logger])
    
    return model, fit


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
    
    plt.plot(epochs,accuracy,'b',label='Training accuracy')
    plt.plot(epochs,val_accuracy,'r',label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(prefix + '_2.png',dpi=500)
    plt.show()


# In[ ]:


dense_model, dense_fit = DenseNetwork(X_train,y_train,img_size*img_size*3,working_classes)


# In[ ]:


validation_plots(dense_fit.history,"ANN")
print(dense_model.evaluate(X_test.reshape(X_test.shape[0],-1),y_test))


# In[ ]:




