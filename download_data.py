import json
import urllib.request
import urllib.error
import subprocess
import random
import multiprocessing
import os

images = []
jsons = os.listdir('imaterialist-challenge-furniture-2018')
for jsonn in jsons:
    jsonn_path = 'imaterialist-challenge-furniture-2018' + '/' + jsonn
    jsonn_location = jsonn.split('.', 1)[0]
    with open(jsonn_path) as json_file:
        jsonn = json.load(json_file)
        for image in jsonn['images']:
            images.append((str(image['url']).strip('[]').replace("'", ""), str(image['image_id']), jsonn_location))

base_dir = '/data/'
def download_images(image):
    filename = image[2] + '/' + image[1]
    fullfilename = os.path.join(base_dir, filename)
    if os.path.exists(fullfilename):
        print("already downloaded", image[2], image[0])
        return

    try:
        request = urllib.request.urlopen(image[0], timeout=30)
        with open(fullfilename, 'wb') as f:
            f.write(request.read())

        print('downloading from ', image[2], image[0])
    except:
         print('link broken for ', image[2], image[0])


random.shuffle(images)

pool = multiprocessing.Pool(multiprocessing.cpu_count())
pool.map(download_images, images)
pool.close()
