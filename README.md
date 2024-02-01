# MatCS
## Overview
It is a binary classification model named MatCS, which is designed to automate the prediction of relationships between NMR spectral images (including 1H NMR and 13C NMR) and the molecular structure of target compounds.
## Requirements
Please refer to the requirement.txt for more details.
## Folder Specification

## Data Generation
The NMR spectra data comes from MestRenova software, with predictive capabilities provided by Qingdao Tenglong Technology Co., Ltd., strictly for academic use.
## Run the Code
We provide five versions of our model. They learn the substructure representations using SMILES, ECFP, GCN, GCN_GAT and GCN_GAT_CBAM, respectively.

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
