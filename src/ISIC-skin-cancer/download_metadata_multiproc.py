import os
import requests
import json
from config import config
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import time
import csv


image_list = json.loads(requests.get(f"https://isic-archive.com/api/v1/image?limit={config['num_imgs']}&offset=0&sort=name&sortdir=1&detail=false",headers={'Accept': 'application/json'}).content)
# image_list = json.loads(requests.get(f"https://isic-archive.com/api/v1/image?limit={5}&offset=0&sort=name&sortdir=1&detail=false",headers={'Accept': 'application/json'}).content)
print(image_list)

outputFileName = 'meta'
outputFilePath = os.path.join(config['data_folder'], outputFileName)

image_details = []

def main_run_multiprocessing_batch():
    workers = max(cpu_count()-1,1)

    manager= Manager()

    listManager = manager.list()
    pool = Pool(workers)

    try:

        part_get_clean_pokemon = partial(get_image_multiprocess, listManager, image_list)
    
        for _ in tqdm(pool.imap(part_get_clean_pokemon, list(range(0, len(image_list)))), total=len(image_list)):
            pass

        pool.close()
        pool.join()
    finally:
        pool.close()
        pool.join()
    
    image_details = list(listManager)
    return image_details
    

def download_image_data(image):
    print('Image:',image)
    image_detail = json.loads(requests.get(f"https://isic-archive.com/api/v1/image/{image['_id']}").content)
    return image_detail


def get_image_multiprocess(listManager=None,images_list=None,process=0):
    global image_details
    image = images_list[process]
    image_data = None
    try:
        image_data = download_image_data(image)
        # image_details.append(image_data)
    except Exception as e:
        print(e);
        return None
    
    finally:
        listManager.append(image_data)
        time.sleep(0.5)
        return image_data

if __name__ == "__main__":
    print('Fetching metadata for %s images' % len(image_list))
    
    image_details = main_run_multiprocessing_batch()

    metadataFields = set(
            field
            for image_detail in image_details
            for field in image_detail['meta']['clinical'].keys()
        )
    
    metadataFields = ['isic_id'] + sorted(metadataFields)
    outputFileName = 'meta'


    outputFilePath = os.path.join(config['data_folder'], outputFileName)

    with open(outputFilePath+'.csv', 'w+') as outputStream:
        csvWriter = csv.DictWriter(outputStream, metadataFields)
        csvWriter.writeheader()
        for imageDetail in image_details:
            rowDict = imageDetail['meta']['clinical'].copy()
            rowDict['isic_id'] = imageDetail['name']
            csvWriter.writerow(rowDict)






