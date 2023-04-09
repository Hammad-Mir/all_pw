import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import regularizers
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils#, multi_gpu_model
from keras.optimizers import Adadelta, RMSprop,SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D, Activation, Input

seed = 7
np.random.seed(seed)

def data_loader(path_train):
   train_list0=os.listdir(path_train)
  
   # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(train_list0)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(train_list0):
           
           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)  
               #resize
               #img2=cv2.resize(img, (300,300))
               # Append image to the train data list
               x.append(img)
               # Append class-label corresponding to the image
               y.append(str(label))
    
        
                
   # Convert lists into numpy arrays

   c = list(zip(x,y))
   np.random.shuffle(c)
   x,y = zip(*c)


   x = np.asarray(x)
   y = np.asarray(y)

   return x,y

path_train = '/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL3/Data/Patched_Train/'

path_test = '/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL3/Data/Patched_Test/'

input_shape = (100, 100, 3)

num_classes = 2

X_train,y_train = data_loader(path_train)
X_test,y_test = data_loader(path_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.
X_test = X_test / 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(300,300,3), kernel_size=(11,11), strides=(4,4), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(2,2), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

#Dropout layer
#model.add(Dropout(0.5))

#Flatten
model.add(Flatten())

model.add(Dropout(0.5))

# 6th FC Layer
model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Dropout(0.5))

# 7th FC Layer
model.add(Dense(4096))
model.add(Activation('relu'))

# 6th FC Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()