from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import json
import os
from PIL import Image
import imgaug.augmenters as iaa
from sklearn.utils import shuffle

base_dir = '/data/'

class Generator(Sequence):
    def __init__(self, data, batch_size, dim_resize, augmentation = False):
        self.x, self.y = data
        self.batch_size = batch_size
        self.dim_resize = dim_resize
        self.augmentation = augmentation

        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.OneOf([
                 iaa.Affine(rotate=(-45, 45)),
                 iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
             ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 2.0)),
                iaa.AverageBlur(k=((5, 11), (1, 3)))
             ]),
        ])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        x_images = []
        y_targets = []

        for file_name, y in zip(batch_x, batch_y):
            try:
                img = np.asarray(Image.open(file_name).convert('RGB'))
                img = cv2.resize(img, dsize=(self.dim_resize, self.dim_resize))
                x_images.append(img)
                y_targets.append(y)
            except Exception as e:
                print('problem with file', file_name)
                continue

        y_targets = np.array(y_targets)
        if self.augmentation == True:
            x_images = self.augment(x_images)

        x_images = np.array(x_images) / 255.

        return x_images, np.array(to_categorical(y_targets, num_classes=128))

    def augment(self, images):
        return self.seq.augment_images(images)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)


def get_data(folder):
    x_set = []
    y_set = []
    with open(base_dir + folder + '.json') as json_file:
        jsonn = json.load(json_file)
        for item in jsonn['annotations']:
            x_set.append('data/' + folder + '/' + item['image_id'])
            y_set.append(item['label_id'][0] - 1)
    return x_set, y_set

