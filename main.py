import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, LeakyReLU, BatchNormalization, SpatialDropout2D
from generator import Generator, get_data
from sklearn.utils import class_weight
import numpy as np
from datetime import datetime
import os

def make_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = input_shape, padding = 'same'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.1))
    model.add(LeakyReLU())

    model.add(Conv2D(32, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3,3), padding = 'same',))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='softmax'))
    return model

model = make_model((256, 256, 3))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

batch_size = 16
training_generator = Generator(get_data('train'), batch_size, 256, augmentation = True)
validation_generator = Generator(get_data('validation'), batch_size, 256)
checkpoint_path = './checkpoints/' + datetime.now().strftime("%Y-%m-%d-%H:%M")

os.makedirs(checkpoint_path, exist_ok=True)

model.fit_generator(generator = training_generator,
    validation_data = validation_generator,
    epochs = 100,
    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./logs/' + datetime.now().strftime("%Y-%m-%d-%H:%M")),
        keras.callbacks.ModelCheckpoint(checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    ],
    class_weight = class_weight.compute_class_weight('balanced', np.unique(training_generator.y), training_generator.y)
)
