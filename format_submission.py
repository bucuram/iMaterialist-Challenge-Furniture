import numpy as np
import os
import json
from PIL import Image
import cv2
import pandas as pd
json_path = '/imaterialist-challenge-furniture-2018/test.json'
images_path = '/data/test/'
images_name = [int(os.path.splitext(filename)[0]) for filename in os.listdir(images_path)]
images = []
with open(json_path) as json_file:
        jsonn = json.load(json_file)
        images= [image['image_id'] for image in jsonn['images'] if int(image['image_id']) in images_name]

data = pd.read_csv("/submission.csv")
df = pd.read_csv('sample_submission_randomlabel.csv')

ids = set(df['id'])
ids2 = set(data['id'])

for item in df['id']:
    if item in data['id']:
        df.at[item-1, 'predicted'] = data.at[item-1, 'predicted']

df.to_csv('/submission.csv', index=False)
