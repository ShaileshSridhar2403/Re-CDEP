from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
import numpy as np
from skimage.morphology import square,dilation
from tqdm import tqdm
from PIL import Image
import os
from config import config
import cd




data_path = config["data_folder"]
processed_path = os.path.join(data_path, "processed")
benign_path = os.path.join(processed_path, "no_cancer")
malignant_path = os.path.join(processed_path, "cancer")
feature_path = os.path.join(data_path, "calculated_features")
segmentation_path = os.path.join(data_path, "segmentation")
os.makedirs(feature_path,exist_ok = True)



model = VGG16(weights='imagenet', include_top=True)

print(model.summary())

list_of_image_names = os.listdir(benign_path)

mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

img_features = np.empty((len(list_of_image_names), 25088))
cd_features = -np.ones((len(list_of_image_names), 2, 25088))
avg_layer = torch.nn.AdaptiveAvgPool2d((7,7))  #NEED TO IMPLEMENT
my_square = square(20)
for i in tqdm(range(len(list_of_image_names))):
        img = Image.open("full image path")
        img = ((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2)[None,:]
        img = ((np.asarray(img)/255.0 -mean)/std)
        # img.close()
        #img_features[i] = avg_layer(model.features(img_torch)).view(-1).cpu().numpy() 
        
        if os.path.isfile(os.path.join(segmentation_path, list_of_image_names[i])):
            seg = Image.open(os.path.join(segmentation_path, list_of_image_names[i]))
            blob =  dilation((np.asarray(seg)[:,:, 0] > 100).astype(np.uint8),my_square).astype(np.float32)
            
            rel, irrel =cd.cd_vgg_features(blob, img, model)
            cd_features[i, 0] = np.array(rel[0])
            cd_features[i, 1] = np.array(irrel[0])


with open(os.path.join(feature_path, "not_cancer.npy"), 'wb') as f:
    np.save(f, img_features)
with open(os.path.join(feature_path, "not_cancer_cd.npy"), 'wb') as f:
    np.save(f, cd_features)


list_of_img_names = os.listdir(malignant_path)
img_features = np.empty((len(list_of_img_names), 25088))
with torch.no_grad():  #CAN I JUST REMOVE THIS
    for i in tqdm(range(len(list_of_img_names))):
        img = Image.open(os.ath.join(malignant_path, list_of_img_names[i]))
        img = ((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2)[None,:] #WHAT DOES THIS DO
        # img.close()
        img_features[i] = avg_layer(model.features(img)).view(-1).cpu().numpy()
with open(os.path.join(feature_path, "cancer.npy"), 'wb') as f:
    np.save(f, img_features)
          

