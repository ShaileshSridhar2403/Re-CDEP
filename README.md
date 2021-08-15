# Deep Explanation Penalization (CDEP)
This repository is a reimplementation of [deep-explanation-penalization](https://github.com/laura-rieger/deep-explanation-penalization) in
`TensorFlow 2.4`.

# Getting Started

## Installation of Packages

Create a virtual environment and install all python packages using
```
pip install requirements.txt
```

## Isic-skin-cancer
Navigate into the ISIC skin cancer directory
```
cd isic-skin-cancer/ISIC-skin-cancer/ 
```

#### Dataset Download and Preprocessing <br>

Download The dataset along with metadata and preprocess

```
python 00_download_metadata.py
python 01_download_images_multiproc.py
python 02_sort_images.py

```

Calculate CD features after propagating through the main body of VGG-16 

```
python 03_calculate_pretrained.py
```

#### Training and Validation 
```
python train_CDEP.py
```

## Stanford Sentiment Dataset <br>

Navigate into the text directory

```
cd text/
```
#### Dataset Download and Preprocessing <br>

Download the Glove embeddings 
```
python download_glove.py

```
Create the random variant of the SST dataset

```
python 00_make_decoy.py 
```
Create the gender variant of the SST dataset<br>

```
python 01_make_gender.py

```
Create the biased variant of the SST dataset
```
python 03_make_bias.py

```

#### Training and Validation
```
python train_all.py
```
Trains and records the results of all experiments<br>

## DecoyMNIST <br>

Navigate to the DecoyMNIST directory
```
cd mnist/DecoyMNIST/
```

Download and process the dataset
```
python 00_make_data.py
```

Train the dataset
```
python 01_train_all.py
```



