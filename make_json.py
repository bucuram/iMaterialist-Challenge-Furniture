import json
import os

jsons = os.listdir('/imaterialist-challenge-furniture-2018')
img_dir = '/data/'

for jsonn in jsons[1:]:
    anno = {'annotations': []}
    jsonn_path = 'imaterialist-challenge-furniture-2018' + '/' + jsonn
    jsonn_location = jsonn.split('.', 1)[0]
    images = os.listdir(img_dir + '/' + jsonn_location)
    with open(jsonn_path) as json_file:
        jsonn = json.load(json_file)
    for image in images:
        anno['annotations'].append({'image_id': image , 'label_id': [annotation['label_id'] for annotation in jsonn['annotations'] if str(annotation['image_id']) == image]})
        print('appending', image)

    with open(img_dir + jsonn_location + '.json', 'wt') as json_new_file:
        json.dump(anno, json_new_file)
