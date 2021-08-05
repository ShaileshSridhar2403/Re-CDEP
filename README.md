# Deep Explanation Penalization (CDEP)
This repository is a reimplementation of [deep-explanation-penalization](https://github.com/laura-rieger/deep-explanation-penalization) in
`TensorFlow 2.4`.

# Getting Started
## Isic-skin-cancer

`cd isic-skin-cancer/ISIC-skin-cancer/` <br>
Data processing <br>
run `python 00_download_metadata.py` <br>
run `python 01_download_images_multiproc.py` <br>
run `python 02_sort_images.py`<br>
run `python 03_calculate_pretrained.py`<br>

Training and Testing <br>
run `python train_CDEP.py`<br>

## Stanford Sentiment Dataset <br>
`cd text/`<br>
Data processing <br>
run `download_glove.py` downloads the glove embeddings <br>
run `python 00_make_decoy.py` creates the random varinat of the SST dataset<br>
run `python 01_make_gender.py` creates the gender varinat of the SST dataset<br>
run `python 03_make_bias.py` creates the biased varinat of the SST dataset<br>
run `python train_all.py` trains and records the results of all experiments<br>

## DecoyMNIST <br>

`cd mnist/DecoyMNIST/` <br>

run `python 00_make_data.py` <br>
run `python 01_train_all.py` <br>


run all the commands in the given order


