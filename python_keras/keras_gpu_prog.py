
# https://www.tensorflow.org/guide/gpu#allowing_gpu_memory_growth

# -v /media/lukas/TeraTest/temp_data/alltours:/data
sudo docker start -ai fc0697014ad5

# >>> läuft!

import tensorflow as tf

# 2060 Su
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)
# END  2060 Su

# [opt] disable warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# import os
# from PIL import Image
# import numpy as np


from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
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

# training_data_path = '/data'

# paperspace P4000/P5000
# training_data_path = '/data/BigSetFull'
# training_data_path = '/data/small'
# training_data_path = '/data/medium'

training_data_path = '/data/train'


train_generator=train_datagen.flow_from_directory(training_data_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size
step_size_train


model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   # use_multiprocessing=True,
                   # workers = 8,
                   # epochs = 10)
                   epochs = 2)
                   # epochs = 30)
                   # epochs = 100)



## Test the model
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    # directory='./test/',
    # directory='/storage/test/',
    directory='/data/test/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode=None,
    shuffle=False
)


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
                            steps=STEP_SIZE_TEST,
                            verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

# oder als Dict definieren (gleicher Task)
# labels = {0: 'Detailbilder', 1: 'Hauptbilder', 2: 'Zimmerbilder'}

predictions = [labels[k] for k in predicted_class_indices]


import pandas as pd

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

# save to csv
results.to_csv("results.csv",index=False)

