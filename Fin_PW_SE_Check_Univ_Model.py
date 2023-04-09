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
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Input, multiply

seed = 7
np.random.seed(seed)

def data_loader(path_train):

   x=[]
   #y=[]
   z=[]
   
   # Loading training data
   
   images=os.listdir(path_train)
   for elem in images:
    path=path_train+'/'+str(elem)
    # Read the image form the directory
    img = cv2.imread(path)
    #resize
    #img2=cv2.resize(img, (300,300))
    # Append image to the train data list
    x.append(img)
    # Append class-label corresponding to the image
    #y.append(int(label))
    z.append(str(elem))
   

   x = np.asarray(x)
   #y = np.asarray(y)
   z = np.asarray(z)

   return x,z

path_test = '/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL3/Data/Patched_Final'

input_shape = (100, 100, 3)

num_classes = 2

X_test, names = data_loader(path_test)

X_test = X_test.astype('float32')
X_test = X_test / 255.
#y_test = np_utils.to_categorical(y)

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
    
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_3)

    se_1 = csse_block(pool1, 32)
  
    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1", kernel_regularizer=regularizers.l2(0.02))(se_1)
    Econv2_1 = BatchNormalization()(Econv2_1)

    Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)

    Econv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv3", kernel_regularizer=regularizers.l2(0.02))(Econv2_2)
    Econv2_3 = BatchNormalization()(Econv2_3)
  
    pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

    se_2 = csse_block(pool2, 64)

    Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1", kernel_regularizer=regularizers.l2(0.02))(se_2)
    Econv3_1 = BatchNormalization()(Econv3_1)

    Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
  
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

    se_3 = csse_block(pool3, 128)
    
    Econv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv1", kernel_regularizer=regularizers.l2(0.02))(se_3)
    Econv4_1 = BatchNormalization()(Econv4_1)

    Econv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv4_1)
    Econv4_2 = BatchNormalization()(Econv4_2)
    
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block4_pool1")(Econv4_2)

    se_4 = csse_block(pool4, 256)

    Econv5_1 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv1", kernel_regularizer=regularizers.l2(0.02))(se_4)
    Econv5_1 = BatchNormalization()(Econv5_1)

    Econv5_2 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv5_1)
    Econv5_2 = BatchNormalization()(Econv5_2)

    pool5 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block5_pool1")(Econv5_2)

    se_5 = csse_block(pool5, 384)

    Econv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv1", kernel_regularizer=regularizers.l2(0.02))(se_5)
    Econv6_1 = BatchNormalization()(Econv6_1)
    
    Econv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv6_1)
    Econv6_2 = BatchNormalization()(Econv6_2)
    
    pool6 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block6_pool1")(Econv6_2)

    se_6 = csse_block(pool6, 512)

    #encoded = Model(inputs = input_img, outputs = pool5 )

    encoded = Model(inputs = input_img, outputs = se_6 )

    return encoded

input_img = Input(shape=input_shape)

encoded = Encoder(input_img)

mod = Sequential()

mod.add(encoded)

mod.add(Flatten())

#mod.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

#mod.add(Dropout(0.3))

mod.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

mod.add(Dropout(0.3))

mod.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.02)))

mod.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0002) ,metrics=['accuracy'])

mod.summary()

mod.load_weights('./Weights/PW_SE_Univ_Model.h5')

predict = mod.predict(X_test, batch_size=1)

y_pred = np.argmax(predict, axis=1)

FC = open('Final_SE_Patch_Check.txt', "w")

#n=np.array([])

for i in range(len(y_pred)):
  
  FC.write('Pred Value : ' + str(y_pred[i]) + '\t' + str(names[i]) + '\n')
  
  #np.save('Fail_Case_Names.npy', n)

FC.close()