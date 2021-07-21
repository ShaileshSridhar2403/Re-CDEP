from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
import numpy as np
from skimage.morphology import square
from tqdm import tqdm
from PIL import Image

def avg_feature():
    #Implement equivalent AdaptiveAveragePool2d in keras
    pass
def calculate_vgg_features(vgg_model,input_x):
    vgg_conv_output = vgg_model.get_layer("block5_conv3")
    feature_model = Model(vgg_model.input,output=vgg_conv_output)
    return feature_model.predict(input_x)

model = VGG16(weights='imagenet', include_top=True)

print(model.summary())

list_of_image_names = []

mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

img_features = np.empty((len(list_of_image_names), 25088))
cd_features = -np.ones((len(list_of_image_names), 2, 25088))
#avg_layer = torch.nn.AdaptiveAvgPool2d((7,7))
my_square = square(20)
for i in tqdm(range(len(list_of_image_names))):
        img = Image.open("full image path")
        #img_torch = torch.from_numpy(((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2))[None,:].cuda().float()
        img = ((np.asarray(img)/255.0 -mean)/std)
        # img.close()
        #img_features[i] = avg_layer(model.features(img_torch)).view(-1).cpu().numpy() 
        
        if os.path.isfile(oj(segmentation_path, list_of_img_names[i])):
            seg = Image.open(oj(segmentation_path, list_of_img_names[i]))
            blob =  dilation((np.asarray(seg)[:,:, 0] > 100).astype(np.uint8),my_square).astype(np.float32)
            
            rel, irrel =cd.cd_vgg_features(blob, img_torch, model)
            cd_features[i, 0] = rel[0].cpu().numpy()
            cd_features[i, 1] = irrel[0].cpu().numpy()

