# -*- coding: utf-8 -*-

from keras.utils import np_utils

import os
import keras
from keras.models import Sequential
from keras.layers import Convolution2D  # Convolution step to add conv layers
from keras.layers import MaxPooling2D # helps to add pooling layers
from keras.layers import Flatten  # takes above layers n converts into large feature vector
from keras.layers import Dense   # full connection

# Initialize the CNN
classifier = Sequential() # classify the 

# Step 1 Convolution
classifier.add(Convolution2D(64,(3,3), input_shape = (32,32,3), activation ='relu')) # 32 no of 3x3 filters(feauture detectors)

# Step 2 MaxPooling
classifier.add(MaxPooling2D(pool_size = [2,2]))

# Adding layer 2
classifier.add(Convolution2D(64,(3,3), activation ='relu'))
classifier.add(Convolution2D(64,(3,3), activation ='relu')) # 32 no of 3x3 filters(feauture detectors)
classifier.add(MaxPooling2D(pool_size = [2,2]))

# Step 3 Flattening
classifier.add(Flatten())

# Step 4 Full connection
classifier.add(Dense(activation = 'relu', units =512)) # no of nodes in hidden layer
classifier.add(Dense(activation = 'sigmoid',units =4))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Compiling CNN
classifier.compile(optimizer = opt,loss = 'categorical_crossentropy', metrics =['accuracy'])

# image augmentation 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=1,
        validation_data=test_set,
        validation_steps=800)

# Making Predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('1.jpg',target_size=(32,32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
