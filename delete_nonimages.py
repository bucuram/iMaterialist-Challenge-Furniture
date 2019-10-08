import filetype
import os
import subprocess
import multiprocessing

folders = os.listdir('/home/cosmadrian/Desktop/iMaterialist Challenge/data')
base_dir = '/home/cosmadrian/Desktop/iMaterialist Challenge/data/'
def delete_files(folder):
    files = [f for f in os.listdir(base_dir + folder)]
    for_deletion = [
        base_dir + folder + '/' + img
        for img in files
            if filetype.guess(base_dir + folder + '/' + img)is None or
                filetype.guess(base_dir + folder + '/' + img).mime.split('/', 1)[0] != 'image'
    ]
    print(folder, len(for_deletion))

    for file in for_deletion:
        os.remove(file)



for folder in folders:
    delete_files(folder)


