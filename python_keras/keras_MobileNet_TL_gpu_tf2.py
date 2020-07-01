
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

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.


x=base_model.output
x=GlobalAveragePooling2D()(x)

# model 1
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

# model 2
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(3,activation='softmax')(x)


model=Model(inputs=base_model.input,outputs=preds)

# Print a summary representation of your model
model.summary()


for layer in model.layers[:87]:
    layer.trainable=False


for layer in model.layers[87:]:
    layer.trainable=True


model.summary()


# train_dir = '/data'

# paperspace P4000/P5000
# train_dir = '/data/BigSetFull'
# train_dir = '/data/small'
# train_dir = '/data/medium'

# local path: BigSetFull
train_dir = '/data/train'
validation_dir = '/data/validation'
test_dir = '/data/test'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['acc'])


step_size_train = train_generator.n//train_generator.batch_size
step_size_train


history = model.fit_generator(
               train_generator,
               steps_per_epoch = step_size_train,
               # epochs = 1,
               epochs = 10,
               # epochs = 100,
               validation_data=validation_generator,
               validation_steps=47,
               verbose=1)


# ca. 188 sek pro Epoche auf Ryzen
# Epoch 1/1 93/93 - 188s 2s/step - loss: 0.5926 - acc: 0.7618 - val_loss: 0.6895 - val_acc: 0.7160

# model1: ca. 60 sek pro Epoche auf 2060 Su
# Epoch 7/10 93/93  - 59s 634ms/step - loss: 0.2736 - acc: 0.8918 - val_loss: 0.6113 - val_acc: 0.7853
# Epoch 10/10 93/93 - 59s 634ms/step - loss: 0.1869 - acc: 0.9292 - val_loss: 0.8205 - val_acc: 0.7487

# model2: ca. 60 sek pro Epoche auf 2060 Su
# Epoch 3/10 93/93  - 61s 651ms/step - loss: 0.3742 - acc: 0.8518 - val_loss: 0.5345 - val_acc: 0.7747
# Epoch 10/10 93/93 - 58s 629ms/step - loss: 0.1587 - acc: 0.9336 - val_loss: 0.9599 - val_acc: 0.7447



# We can now finally evaluate this model on the test data:

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


test_loss, test_acc = model.evaluate_generator(test_generator, steps=30)

test_loss,
test_acc


