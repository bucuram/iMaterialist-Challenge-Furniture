from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from generator import Generator, get_data
from tensorflow.keras.optimizers import SGD
from datetime import datetime
from sklearn.utils import class_weight
import numpy as np
import os

batch_size = 32
input_tensor = Input(shape=(256, 256, 3))
# base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(128, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training_generator = Generator(get_data('train'), batch_size, 256, augmentation = True)
validation_generator = Generator(get_data('validation'), batch_size, 256)
checkpoint_path = './checkpoints/inceptionresnetv2/' + datetime.now().strftime("%Y-%m-%d-%H:%M")
os.makedirs(checkpoint_path, exist_ok=True)

model.fit_generator(generator = training_generator,
    validation_data = validation_generator,
    epochs = 1,
    callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/inceptionresnetv2/' + datetime.now().strftime("%Y-%m-%d-%H:%M")),
            keras.callbacks.ModelCheckpoint(checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', period=1),
        ],
    class_weight = class_weight.compute_class_weight('balanced', np.unique(training_generator.y), training_generator.y)
)


for epoch in range(1, 51):

    for layer in model.layers[:250 - epoch * 5]:
       layer.trainable = False
    for layer in model.layers[250 - epoch * 5:]:
       layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator = training_generator,
        validation_data = validation_generator,
        epochs = 5,
        initial_epoch=epoch,
        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./logs/inception' + datetime.now().strftime("%Y-%m-%d-%H:%M")),
            keras.callbacks.ModelCheckpoint(checkpoint_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', period=1),
        ],
        class_weight = class_weight.compute_class_weight('balanced', np.unique(training_generator.y), training_generator.y)
    )
