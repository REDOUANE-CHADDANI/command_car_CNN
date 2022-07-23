# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:47:25 2022

@author: REDOUANE-CH
"""

import splitfolders

Database = ('D:\Projet_voiture_cnn\Database')
Data_Splitted = ('D:\Projet_voiture_cnn\data_splitted')

splitfolders.ratio(Database, Data_Splitted , ratio = (0.7 , 0 ,0.3 ))

import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics 
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import numpy as np 

#create path 
train_path = 'D:\Projet_voiture_cnn\data_splitted/train/'
test_path = 'D:\Projet_voiture_cnn\data_splitted/test/'

#resize images
# Normalize pixel values to be between 0 and 1
trdata = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,)
traindata = trdata.flow_from_directory(directory = train_path, target_size=(224,224)) 

tsdata = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,)
testdata = tsdata.flow_from_directory(directory = test_path, target_size=(224,224)) 



#categories
root=pathlib.Path(test_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes) 
print(traindata.class_indices)

#define input image
input_shape =(224,224,3)

#create the network
#Input layer
img_input = Input(shape = input_shape, name='img_input')

#Build the Model
x = Conv2D(32, (3,3), padding='same', activation='relu', name='layer_1') (img_input)
x = Conv2D(64, (3,3), padding='same', activation='relu', name='layer_2') (x)
x = MaxPool2D((2,2), strides = (2,2), name='layer_3') (x)
x = Dropout(0.1) (x)

x = Conv2D(128, (3,3), padding='same', activation='relu', name='layer_4') (x)
x = MaxPool2D((2,2), strides = (2,2), name='layer_5') (x)
x = Dropout(0.1) (x)

x = Flatten(name = 'fc_1') (x)
x = Dense(64, name = 'layer_8') (x)
x = Dropout(0.1) (x)
x = Dense(5, activation= 'sigmoid', name = 'predictions') (x)

#Generate the Model
model = Model(inputs = img_input, outputs = x , name='CNN_classification')

#Print network structure
model.summary()

#Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

#Save The Model
model.save('model_trained.h5')

#start Train/Test
batch_size = 32
hist = model.fit(traindata, steps_per_epoch = traindata.samples//batch_size, 
                 validation_data =testdata, validation_steps = testdata.samples//batch_size,
                 epochs = 5)

plt.plot(hist.history['loss'], label ='train')
plt.plot(hist.history['val_loss'], label ='val')
plt.title('Training & test loss')
plt.legend()
plt.show() 

plt.plot(hist.history['accuracy'], label ='train')
plt.plot(hist.history['val_accuracy'], label ='val')
plt.title('Training & test accuracy')
plt.legend()
plt.show() 

#Confusion Matrix & Pres & Recall & F1-Score
target_names =['Arr', 'AV', 'M-Arr', 'T-D', 'T-G']
label_names =[0,1,2,3,4]

x_pred = model.predict_generator(testdata)
y_pred = np.argmax(x_pred , axis = 1)

cm = confusion_matrix(testdata.classes, y_pred , labels = label_names)

print('Confusion Matrix')
print(confusion_matrix(testdata.classes , y_pred))

print('Classification_Report')
print(classification_report(testdata.classes , y_pred, target_names = target_names ))

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names)
disp = disp.plot(cmap =plt.cm.Blues , values_format = 'g')
plt.show()

