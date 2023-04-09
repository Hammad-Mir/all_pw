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
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input, Conv2DTranspose

seed = 7
np.random.seed(seed)

path_train = '/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL1/Data/Medical_Challange_train/Augmented'

path_test = '/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL2/Test_f'


batch_size = 6

input_shape = (100, 100, 3)

num_classes = 2


def train_generator(path_train, batch_size):

	t_l = os.listdir(path_train)
	no_cla = len(t_l)

	path1=path_train+'/'+str(t_l[0])    # For class 1 (all)
	path2=path_train+'/'+str(t_l[1])    # For class 2 (hem)

	images_all = os.listdir(path1)
	images_hem = os.listdir(path2)

	steps = len(images_all)//(batch_size*2//3)
	c = 0
	i = 0
	j = 0
	c1 = batch_size*2//3
	c2 = batch_size*1//3

	while True:
		x_t = []
		y_t = []
		# Loading 2/3 images from all
		while i < c1:
			path_im1 = path1+'/'+str(images_all[i])
			# Read the image form the directory
			img = cv2.imread(path_im1)
			for a in range(0, 300, 50):
				for b in range(0, 300, 50):
					im = img[a:a+100, b:b+100]
					x_t.append(im)
					y_t.append('0')
			i += 1

		# Loading 1/3 images from hem    
		while j < c2:
			path_im2 = path2+'/'+str(images_hem[j])
			# Read the image form the directory
			img = cv2.imread(path_im2)
			for a in range(0, 300, 50):
				for b in range(0, 300, 50):
					im = img[a:a+100, b:b+100]
					x_t.append(im)
					y_t.append('1')
			j += 1

		com = list(zip(x_t,y_t))
		np.random.shuffle(com)
		x_t,y_t = zip(*com)

		x_t = np.asarray(x_t)
		y_t = np.asarray(y_t)
		x_t = x_t.astype('float32')
		x_t = x_t / 255
		y_t = np_utils.to_categorical(y_t)

		c1 += batch_size*2//3
		c2 += batch_size*1//3

		yield (x_t, y_t)
        #return x_t, y_t

gen = train_generator(path_train, batch_size)


def data_loader(path_test):
   test_list0=os.listdir(path_test)
  
   # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(test_list0)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(test_list0):
           
           path_1=path_test+'/'+str(elem)
           images=os.listdir(path_1)
           for elem2 in images:
               path_2=path_1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path_2)  
               for a in range(0, 300, 50):
                for b in range(0, 300, 50):
                    im = img[a:a+100, b:b+100]
                    x.append(im)
                    y.append(str(label))

   ci = list(zip(x,y))
   np.random.shuffle(ci)
   x,y = zip(*ci)

   x = np.asarray(x)
   y = np.asarray(y)

   return x,y

#print(X_train.shape)
#print(y_train.shape)

#X_train,y_train = data_loader(path_train)
#X_train = X_train.astype('float32')
#X_train = X_train / 255.
#y_train = np_utils.to_categorical(y_train)

X_test,y_test = data_loader(path_test)

print(X_test.shape)
print(y_test.shape)

X_test = X_test.astype('float32')
X_test = X_test / 255.
y_test = np_utils.to_categorical(y_test)

def Encoder(input_img):

    Econv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name = "block1_conv1", kernel_regularizer=regularizers.l2(0.02))(input_img)
    Econv1_1 = BatchNormalization()(Econv1_1)
  
    Econv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same',  name = "block1_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)

    Econv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same',  name = "block1_conv3", kernel_regularizer=regularizers.l2(0.02))(Econv1_2)
    Econv1_3 = BatchNormalization()(Econv1_3)
    
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_3)
  
    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1", kernel_regularizer=regularizers.l2(0.02))(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)

    Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)

    Econv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv2_2)
    Econv2_3 = BatchNormalization()(Econv2_3)
  
    pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

    Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1", kernel_regularizer=regularizers.l2(0.02))(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)

    Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
  
    pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)
    
    Econv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv1", kernel_regularizer=regularizers.l2(0.02))(pool3)
    Econv4_1 = BatchNormalization()(Econv4_1)

    Econv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name = "block4_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv4_1)
    Econv4_2 = BatchNormalization()(Econv4_2)
    
    pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block4_pool1")(Econv4_2)

    Econv5_1 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv1", kernel_regularizer=regularizers.l2(0.02))(pool4)
    Econv5_1 = BatchNormalization()(Econv5_1)

    Econv5_2 = Conv2D(384, (3, 3), activation='relu', padding='same', name = "block5_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv5_1)
    Econv5_2 = BatchNormalization()(Econv5_2)

    pool5 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block5_pool1")(Econv5_2)

    Econv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv1", kernel_regularizer=regularizers.l2(0.02))(pool5)
    Econv6_1 = BatchNormalization()(Econv6_1)
    
    Econv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name = "block6_conv2", kernel_regularizer=regularizers.l2(0.02))(Econv6_1)
    Econv6_2 = BatchNormalization()(Econv6_2)
    
    pool6 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block6_pool1")(Econv6_2)

    #encoded = Model(inputs = input_img, outputs = pool5 )

    encoded = Model(inputs = input_img, outputs = pool6 )

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

#mod.load_weights('./Weights/Univ_Model.h5')

#mod.load_weights('/home/biometric/Dr_Aditya_Nigam/AAL_Intern/ALL2/Pr_model_analysis/pr.h5')

cp = ModelCheckpoint(filepath = './Weights/PW_Gen_Univ_Model.h5', verbose = 1, save_best_only = True, monitor='val_acc')

try:
  #mod.fit(X_train, y_train, batch_size=100, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks = [cp])
  mod.fit_generator(generator=(gen),
                  steps_per_epoch=3120,
                  validation_data=(X_test, y_test),
                  validation_steps=1,
                  epochs=100,
                  callbacks = [cp])

except KeyboardInterrupt:
  mod.load_weights('./Weights/PW_Gen_Univ_Model.h5')

  scores = mod.evaluate(X_test, y_test, verbose=2)

  print("Accuracy 1: %.2f%%" % (scores[1]*100))

mod.load_weights('./Weights/PW_Gen_Univ_Model.h5')

scores = mod.evaluate(X_test, y_test, verbose=2)

print("Accuracy 2: %.2f%%" % (scores[1]*100))