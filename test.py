import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import numpy as np
import os
import json
from PIL import Image
import cv2
import pandas as pd

dim_resize = 256
json_path = '/imaterialist-challenge-furniture-2018/test.json'
images_path = '/data/test/'
images_name = [int(os.path.splitext(filename)[0]) for filename in os.listdir(images_path)]
images = []
with open(json_path) as json_file:
        jsonn = json.load(json_file)
        images= [image['image_id'] for image in jsonn['images'] if int(image['image_id']) in images_name]

x_images = []
predictions = []
# model = load_model('./checkpoints/2019-08-28-23:39/weights.13-1.74.hdf5') #first submission
# model = load_model('./checkpoints/inception/2019-08-30-10:44/weights.24-0.89.hdf5') #second submission. inception
model = load_model('./checkpoints/inceptionresnetv2/2019-09-01-16:41/weights.05-1.80.hdf5')
for i, image in enumerate(images):
    try:
        image_path = 'data/test/'+ str(image) + '.jpg'
        img = np.asarray(Image.open(image_path).convert('RGB'))
        img = cv2.resize(img, dsize=(dim_resize, dim_resize))
        p = np.argmax(model.predict(np.array([img / 255.]))[0])
        predictions.append((image, p))
        # x_images.append(img)
        print(i, len(images), image, p)
    except Exception as e:
        print('problem with file', image)
        continue

submission = pd.DataFrame([['id', 'predicted']])
for image, pred in predictions:
    submission = submission.append([[image, pred + 1]])
submission.to_csv('/submission.csv', index=False, header=None)
