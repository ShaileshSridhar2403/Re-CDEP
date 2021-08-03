import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.morphology import dilation, square
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tqdm import tqdm

from config import config

sys.path.append('../../src')
import cd


def check_and_convert_to_NHWC(img):
    # print(img.shape)
    if img.shape[-1] != 3:
        img = tf.transpose(img, [0, 2, 3, 1])
        # print("converted shape",img.shape)
    return img


def features(vgg_model, img):
    features_output = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    return features_output.predict(img)


data_path = config["data_folder"]
processed_path = os.path.join(data_path, "processed")
benign_path = os.path.join(processed_path, "no_cancer")
malignant_path = os.path.join(processed_path, "cancer")
feature_path = os.path.join(data_path, "calculated_features")
segmentation_path = os.path.join(data_path, "segmentation")
os.makedirs(feature_path, exist_ok=True)

model = VGG16(weights='imagenet', include_top=True)

print(model.summary())

list_of_image_names = os.listdir(benign_path)
len1 = len(list_of_image_names)
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

img_features = np.empty((len(list_of_image_names), 25088))
cd_features = -np.ones((len(list_of_image_names), 2, 25088))
# avg_layer = torch.nn.AdaptiveAvgPool2d((7,7))  #NEED TO IMPLEMENT
my_square = square(20)  # should this be 224/299 * 20
for i in tqdm(range(len(list_of_image_names))):
    full_image_path = os.path.join(benign_path, list_of_image_names[i])
    img = Image.open(full_image_path)
    img = img.resize((224, 224))
    # pdb.set_trace()
    img = ((np.asarray(img) / 255.0 - mean) / std).swapaxes(0, 2).swapaxes(1, 2)[None, :]
    # img_orig = img
    # img = check_and_convert_to_NHWC(img)
    img_features[i] = features(model, check_and_convert_to_NHWC(img))

    if os.path.isfile(os.path.join(segmentation_path, list_of_image_names[i])):
        seg = Image.open(os.path.join(segmentation_path, list_of_image_names[i]))
        # print("seg",np.array(seg).shape)
        seg = seg.resize((224, 224))
        # print("seg2",np.array(seg).shape)
        # print(seg.shape())
        # try:
        #     print("seg shape",seg.shape())
        # except Exception as e:
        #     print(f"SKIPPPING due to {e}")
        # continue
        blob = dilation((np.asarray(seg)[:, :, 0] > 100).astype(np.uint8), my_square).astype(np.float32)
        # print("masked",tf.where(blob,img,tf.zeros(img.shape)))

        rel, irrel = cd.cd_vgg_features(blob, img, model)
        cd_features[i, 0] = np.array(rel[0])
        cd_features[i, 1] = np.array(irrel[0])

    else:
        # print(os.path.join(segmentation_path, list_of_image_names[i]))
        pass

# print(size(cd_features))
with open(os.path.join(feature_path, "not_cancer.npy"), 'wb') as f:
    np.save(f, img_features)
with open(os.path.join(feature_path, "not_cancer_cd.npy"), 'wb') as f:
    np.save(f, cd_features)

list_of_img_names = os.listdir(malignant_path)
img_features = np.empty((len(list_of_img_names), 25088))
for i in tqdm(range(len(list_of_img_names))):
    img = Image.open(os.path.join(malignant_path, list_of_img_names[i]))
    img = img.resize((224, 224))
    img = ((np.asarray(img) / 255.0 - mean) / std).swapaxes(0, 2).swapaxes(1, 2)[None, :]  # WHAT DOES THIS DO
    img = check_and_convert_to_NHWC(img)
    # img.close()
    img_features[i] = features(model, img)

with open(os.path.join(feature_path, "cancer.npy"), 'wb') as f:
    np.save(f, img_features)

print("Complete Length", len1 + len(img_features))
