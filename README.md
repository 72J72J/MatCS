# MatCS
## Overview
It is a binary classification model named MatCS, which is designed to automate the prediction of relationships between NMR spectral images (including 1H NMR and 13C NMR) and the molecular structure of target compounds.
## Requirements
Please refer to the requirement.txt for more details.
## Folder Specification
data/ folder contains necessary data for train the model.

process_data.py: Code for processing data.

predict.py: Code for predicting the relationships between NMR spectral images and target compounds' structure.

train.py: Train our MatCS Model.

GATencoderattention.py: Code for model definition.

utils.py: Code for metric calculations and some data preparation.
## Data Generation
The NMR spectra data comes from MestRenova software, with predictive capabilities provided by Qingdao Tenglong Technology Co., Ltd., strictly for academic use.
These files are large and you can download it from https://drive.google.com/drive/folders/1Ea6SNo5MmwItuRoaWcDvZECka6gBnUp9?usp=sharing.
## Run the Code
To process the data, use the following command:
```shell
python process_data.py
```
To train the model, use the following command:
```shell
python train.py
```
To evaluate a well-trained model, use the following command:
```shell
python predict.py
```
