import os
import requests
import json
from config import config
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import time


data_path = config["data_folder"]
num_imgs = config["num_imgs"]
#%%
savePath = os.path.join(data_path, 'raw')

if not os.path.exists(savePath):
    os.makedirs(savePath)
start_offset = 0
#%%
def get_image_list(image_batch_ind):
    start_offset = image_batch_ind*50
    image_list = json.loads(requests.get(f"https://isic-archive.com/api/v1/image?limit={50}&offset={start_offset}&sort=name",headers={'Accept': 'application/json'}).content)
    return image_list

def download_image(image):
    imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['name'])
    if os.path.exists(imageFileOutputPath):
        print(f"Image already exists at path {imageFileOutputPath}")
        return

    imageFileResp = requests.get(f"https://isic-archive.com/api/v1/image/{image['_id']}/download")
    imageFileResp.raise_for_status()
    
    print(f"Saving image to {imageFileOutputPath}")
    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
        for chunk in imageFileResp:
            imageFileOutputStream.write(chunk)
    print(f"Downloaded Image with id {image['_id']}")


def get_image_multiprocess(listManager=None,images_list=None,process=0):
    image = images_list[process]
    info = None
    try:
        download_image(image)
    except Exception as e:
        print(e)
    
    finally:
        listManager.append(info)
        time.sleep(0.5)
        return

def main_run_multiprocessing_batch(batch_ind):
    workers = max(cpu_count()-1,1)

    manager= Manager()

    listManager = manager.list()
    pool = Pool(workers)

    try:
        image_list = get_image_list(batch_ind)
        part_get_clean_pokemon = partial(get_image_multiprocess, listManager, image_list)

        for _ in tqdm(pool.imap(part_get_clean_pokemon, list(range(0, len(image_list)))), total=len(image_list)):
            pass
        pool.close()
        pool.join()
    finally:
        pool.close()
        pool.join()
        


# for i in range(int(num_imgs/50)+1):
#     print(f"Downloading set {i} out of {int(num_imgs/50)+1}")
#     # imageList = api.getJson('image?limit=50&offset=' + str(start_offset) + '&sort=name')
#     image_list = json.loads(requests.get(f"https://isic-archive.com/api/v1/image?limit={50}&offset={start_offset}&sort=name",headers={'Accept': 'application/json'}).content)
    
#     print(f'Downloading {len(image_list)} images')
    
#     for image in image_list:
#         print(f"Downloading image with id {image['_id']}")
#         # imageFileResp = api.get('image/%s/download' % image['_id'])
#         imageFileResp = requests.get(f"https://isic-archive.com/api/v1/image/{image['_id']}/download")

#         imageFileResp.raise_for_status()
#         imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['name'])
#         print(f"Saving image to {imageFileOutputPath}")
#         with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
#             for chunk in imageFileResp:
#                 imageFileOutputStream.write(chunk)
#     start_offset +=50
# %%


if __name__ =="__main__":
    custom_interval = input("Would you like to download a custom interval of image batchs?")
    if 'y' in custom_interval:
        start_batch_ind = int(input("Enter start batch indice"))
        end_batch_ind = int(input("Enter end batch indice"))
    else:
        start_batch_ind = 0
        end_batch_ind = int(num_imgs/50)+1

    for i in range(start_batch_ind,end_batch_ind):
        print(f"Downloading set {i} out of {int(num_imgs/50)+1}")
        main_run_multiprocessing_batch(i)