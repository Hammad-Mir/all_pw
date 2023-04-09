import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense , Conv2D, Dropout, Flatten, add, Reshape
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Input, multiply, Concatenate

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

def cse_block(in_block, channels, ratio=16):

    ch = channels
    x_shape = (1, 1, ch)
    x = GlobalAveragePooling2D()(in_block)
    x = Reshape(x_shape)(x)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return multiply([in_block, x])

def sse_block(in_block):

  x = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(in_block)
  x = multiply([in_block, x])
  return x

def csse_block(in_block, channels, ratio=16):

  cse = cse_block(in_block, channels, ratio)
  sse = sse_block(in_block)
  x = add([cse, sse])
  return x

def Encoder(input_img):

    Econv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name = "block1_conv1", kernel_regularizer=regularizers.l2(0.02))(input_img)
    Econv1_1 = BatchNormalization()(Econv1_1)
  
    Econv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same',  name = "block1_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)

    Econv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same',  name = "block1_conv3", kernel_regularizer=regularizers.l2(0.02))(Econv1_2)
    Econv1_3 = BatchNormalization()(Econv1_3)

    se_1 = csse_block(Econv1_3, 32)
    Con_1 = Concatenate()([input_img, se_1])
    
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Con_1)

    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1", kernel_regularizer=regularizers.l2(0.02))(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)

    Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)

    Econv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv3", kernel_regularizer=regularizers.l2(0.02))(Econv2_2)
    Econv2_3 = BatchNormalization()(Econv2_3)

    se_2 = csse_block(Econv2_3, 64)
    Con_2 = Concatenate()([pool1, se_2])
  
    pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Con_2)

    Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1", kernel_regularizer=regularizers.l2(0.02))(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)

    Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)

    se_3 = csse_block(Econv3_2, 128)
    Con_3 = Concatenate()([pool2, se_3])
  
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Con_3)
    
    Econv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv1", kernel_regularizer=regularizers.l2(0.02))(pool3)
    Econv4_1 = BatchNormalization()(Econv4_1)

    Econv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv4_1)
    Econv4_2 = BatchNormalization()(Econv4_2)

    se_4 = csse_block(Econv4_2, 256)
    Con_4 = Concatenate()([pool3, se_4])
    
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block4_pool1")(Con_4)

    Econv5_1 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv1", kernel_regularizer=regularizers.l2(0.02))(pool4)
    Econv5_1 = BatchNormalization()(Econv5_1)

    Econv5_2 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv5_1)
    Econv5_2 = BatchNormalization()(Econv5_2)

    se_5 = csse_block(Econv5_2, 384)
    Con_5 = Concatenate()([pool4, se_5])

    pool5 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block5_pool1")(Con_5)

    Econv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv1", kernel_regularizer=regularizers.l2(0.02))(pool5)
    Econv6_1 = BatchNormalization()(Econv6_1)
    
    Econv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv6_1)
    Econv6_2 = BatchNormalization()(Econv6_2)

    se_6 = csse_block(Econv6_2, 512)
    Con_6 = Concatenate()([pool5, se_6])
    
    pool6 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block6_pool1")(Con_6)

    #encoded = Model(inputs = input_img, outputs = pool5 )

    encoded = Model(inputs = input_img, outputs = pool6 )

    return encoded

input_img = Input(shape=input_shape)

encoded = Encoder(input_img)

mod = Sequential()

mod.add(encoded)

mod.add(Flatten())

#mod.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

mod.add(Dropout(0.3))

mod.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

mod.add(Dropout(0.3))

mod.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.02)))

mod.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0002) ,metrics=['accuracy'])

mod.summary()

cp = ModelCheckpoint(filepath = './Weights/PW_SE_2_Univ_Model.h5', verbose = 1, save_best_only = True, monitor='val_acc')

try:
  mod.fit(X_train, y_train, batch_size=150, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks = [cp])

except KeyboardInterrupt:
  
  mod.load_weights('./Weights/PW_SE_2_Univ_Model.h5')

  scores = mod.evaluate(X_test, y_test, verbose=2)

  print("Accuracy 1: %.2f%%" % (scores[1]*100))

mod.load_weights('./Weights/PW_2_SE_Univ_Model.h5')

scores = mod.evaluate(X_test, y_test, verbose=2)

print("Accuracy 2: %.2f%%" % (scores[1]*100))