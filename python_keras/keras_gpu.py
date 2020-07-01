
# conda activate tfgpu

# conda activate tfgpu1.14


# tests
import tensorflow as tf

import tensorflow

from tensorflow import keras


# >>> läuft!

import tensorflow as tf

import os

from PIL import Image

import numpy as np

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, model_from_yaml
from tensorflow.keras.optimizers import Adam


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

# Print a summary representation of your model
model.summary()


for layer in model.layers[:87]:
	layer.trainable=False


for layer in model.layers[87:]:
	layer.trainable=True


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies


# local path: BigSetFull
# training_data_path = '/media/lukas/TeraTest/temp/alltours/BigSetFull'

training_data_path = '/data'

# paperspace P4000
# training_data_path = '/data/BigSetFull'


train_generator=train_datagen.flow_from_directory(training_data_path,
	target_size=(224,224),
	color_mode='rgb',
	batch_size=16,
	class_mode='categorical',
	shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size


model.fit_generator(generator=train_generator,
	steps_per_epoch=step_size_train,
	# use_multiprocessing=True,
	# workers = 8,
	epochs = 10)
	# epochs = 30)
	# epochs = 100)



