import csv
import numpy as np
from tqdm import tqdm
import scipy.misc
from shutil import copyfile
import os
import json
import csv
from config import config
from PIL import Image
import cv2

def resize_and_save(my_list, my_folder):
    for i,file_name in tqdm(enumerate(my_list)):
        try:
            img = Image.open(os.path.join(img_path, file_name))
            test = np.asarray(img)
            test_new = cv2.resize(test, (299, 299, 3))
            cv2.imwrite(os.path.join(my_folder, file_name), test_new)
        except:
            print(file_name)

data_path = config['data_folder']

img_path = os.path.join(data_path, "raw")
processed_path = os.path.join(data_path, "processed")
segmentation_path = os.path.join(data_path, "segmentation")
benign_path = os.path.join(processed_path, "no_cancer")
malignant_path = os.path.join(processed_path, "cancer")
os.makedirs(processed_path,exist_ok = True)
os.makedirs(benign_path,exist_ok = True)
os.makedirs(segmentation_path,exist_ok = True)
os.makedirs(malignant_path,exist_ok = True)

list_of_meta = []
with open(os.path.join(data_path, "meta.csv"), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader)
    for row in spamreader:
        list_of_meta.append(row) 

list_benign_files = []
list_mal_files = []
for line in list_of_meta[1:]:
    if len(line) > 0:
        if line[3] == 'benign':
            list_benign_files.append(line[0] + ".jpg")
        elif line[3] == 'malignant':
            list_mal_files.append(line[0] + ".jpg")

resize_and_save(list_mal_files,malignant_path)
resize_and_save(list_benign_files,benign_path)
    


