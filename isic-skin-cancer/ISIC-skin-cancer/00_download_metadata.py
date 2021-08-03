import csv
import json
import os

import requests
from tqdm import tqdm

from config import config

image_list = json.loads(requests.get(
    f"https://isic-archive.com/api/v1/image?limit={config['num_imgs']}&offset=0&sort=name&sortdir=1&detail=false",
    headers={'Accept': 'application/json'}).content)
print(image_list)

if __name__ == "__main__":
    print('Fetching metadata for %s images' % len(image_list))
    image_details = []
    for image in tqdm(image_list):
        print('Image:', image)
        image_detail = json.loads(requests.get(f"https://isic-archive.com/api/v1/image/{image['_id']}").content)
        image_details.append(image_detail)

    metadataFields = set(
        field
        for image_detail in image_details
        for field in image_detail['meta']['clinical'].keys()
    )

    metadataFields = ['isic_id'] + sorted(metadataFields)
    outputFileName = 'meta'

    outputFilePath = os.path.join(config['data_folder'], outputFileName)

    print('Writing metadata to CSV: %s' % outputFileName + '.csv')
    with open(outputFilePath + '.csv', 'w+') as outputStream:
        csvWriter = csv.DictWriter(outputStream, metadataFields)
        csvWriter.writeheader()
        for imageDetail in image_details:
            rowDict = imageDetail['meta']['clinical'].copy()
            rowDict['isic_id'] = imageDetail['name']
            csvWriter.writerow(rowDict)
