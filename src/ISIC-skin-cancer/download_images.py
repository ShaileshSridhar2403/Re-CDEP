import os
import requests
import json
from config import config
from tqdm import tqdm

data_path = config["data_folder"]
num_imgs = config["num_imgs"]
#%%
savePath = os.path.join(data_path, 'raw')

if not os.path.exists(savePath):
    os.makedirs(savePath)
start_offset = 0
#%%

for i in range(int(num_imgs/50)+1):
    print(f"Downloading set {i} out of {int(num_imgs/50)+1}")
    # imageList = api.getJson('image?limit=50&offset=' + str(start_offset) + '&sort=name')
    image_list = json.loads(requests.get(f"https://isic-archive.com/api/v1/image?limit={50}&offset={start_offset}&sort=name",headers={'Accept': 'application/json'}).content)
    
    print(f'Downloading {len(image_list)} images')
    
    for image in image_list:
        print(f"Downloading image with id {image['_id']}")
        # imageFileResp = api.get('image/%s/download' % image['_id'])
        imageFileResp = requests.get(f"https://isic-archive.com/api/v1/image/{image['_id']}/download")

        imageFileResp.raise_for_status()
        imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['name'])
        print(f"Saving image to {imageFileOutputPath}")
        with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
            for chunk in imageFileResp:
                imageFileOutputStream.write(chunk)
    start_offset +=50