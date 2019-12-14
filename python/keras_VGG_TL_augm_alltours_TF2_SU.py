
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



## main
from tensorflow.keras.applications import VGG16

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os


base_dir = 'T:\\temp_data\\alltours'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# local path Ubuntu/Docker: BigSetFull
train_dir = '/data/train'
validation_dir = '/data/validation'
test_dir = '/data/test'


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


conv_base.summary()


## FC Base
# model naiv
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))


# model FL 1024*1024*512
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))


model.summary()

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))

model.summary()


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)


# https://keras.io/preprocessing/image/

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=30,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=30,
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


# https://keras.io/models/sequential/
# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      # epochs=1,
      # epochs=30,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)


# ca. 150 sek pro Epoche auf Ryzen
# Epoch 2/2 - 149s - loss: 0.4621 - acc: 0.8065 - val_loss: 0.3580 - val_acc: 0.8740
# Epoch 15/15 100/100 - 150s 2s/step - loss: 0.3151 - acc: 0.8625 - val_loss: 0.2466 - val_acc: 0.9010
# (weitere 10 Epochen [15+10]) Epoch 10/10 100/100 - 151s 2s/step - loss: 0.2945 - acc: 0.8720 - val_loss: 0.2365 - val_acc: 0.9070

## alltours: model naiv
# ca. 220 sek pro Epoche auf Ryzen
# Epoch 2/2 100/100 - 224s 2s/step - loss: 0.6262 - acc: 0.7287 - val_loss: 0.5765 - val_acc: 0.7540
# Epoch 15/15 100/100 - 224s 2s/step - loss: 0.4618 - acc: 0.8153 - val_loss: 0.4897 - val_acc: 0.7727

# alltours: model FL 1024*1024*512
# ca. 234 sek pro Epoche auf Ryzen
# Epoch 1/1 100/100 - 234s 2s/step - loss: 0.7356 - acc: 0.6787 - val_loss: 0.5614 - val_acc: 0.7480

## 2060 SU
# alltours: model FL 1024*1024*512
# ca. 66 sek pro Epoche auf 2060 SU
# Epoch 10/10 100/100 - 65s 650ms/step - loss: 0.4708 - acc: 0.8070 - val_loss: 0.4965 - val_acc: 0.7840
# weitere 10 Ep. auf dem gleichen Modell (10+10)
# Epoch 10/10 100/100 - 65s 652ms/step - loss: 0.4386 - acc: 0.8240 - val_loss: 0.4763 - val_acc: 0.7840



model.save('T:\\temp_data\\alltours\\vgg_tl_augm_v1_2ep.h5')

model.save('T:\\temp_data\\alltours\\vgg_tl_augm_v2_17ep.h5')

# ubuntu/docker
model.save('/data/alltours_vgg_tl_augm_ubuntu_20ep.h5')


## Model evaluation


from tensorflow.keras.models import load_model

# https://stackoverflow.com/questions/49195189/error-loading-the-saved-optimizer-keras-python-raspberry

# model = load_model('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_25ep.h5')

model = load_model('/data/alltours_vgg_tl_augm_ubuntu_20ep.h5')

model.summary()



# We can now finally evaluate this model on the test data:

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


test_loss, test_acc = model.evaluate_generator(test_generator, steps=30)

test_loss,
test_acc


# 10 EP. alltours: model FL 1024*1024*512
# 0.659375

# 20 EP. alltours: model FL 1024*1024*512
# 0.709375



