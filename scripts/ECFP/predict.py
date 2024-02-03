import torch
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import os
from sklearn import metrics
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from tqdm import tqdm
from sklearn.metrics import auc
from utils import CaptionDataset
import seaborn as sns


def main(args):
    # Load model
    try:
        print("Loading models: {}".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        encoder = checkpoint['encoder']
    except Exception as e:
        print("Models couldn't be loaded. Aborting")
        print(e)
        exit(0)

    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
    else:
        device = 'cpu'
    encoder = encoder.to(device)
    encoder.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_dataset = CaptionDataset(args.data_dir, 'test', 'test_C',transforms.Compose([normalize]))
    print(len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("There are {} validation data examples".format(len(test_dataset)))

    print("Predicting images")
    y_true = []
    outputs_all = np.array([], dtype=int)
    targets_all = np.array([], dtype=int)
    with torch.no_grad():

        for data in tqdm(test_loader):
            imgs = data[0]
            morgens = data[1]
            targets = data[2]
            imgsC = data[3]
            imgs = imgs.to(device)
            morgens = morgens.to(device)
            imgsC = imgsC.to(device)

            targets = targets.to(device)
            y_true.extend(targets.detach().tolist())
            outputs = encoder(imgs, morgens,imgsC).to(device)
            outputs_all = np.append(outputs_all, outputs.data)
            targets_all = np.append(targets_all, targets.data).tolist()

        new_output_all = []
        for i in range(outputs_all.shape[0]):
            if outputs_all[i] > args.threshold:
                new_output_all.append(1)
            else:
                new_output_all.append(0)
        new_output_all = np.array(new_output_all).tolist()
        print(new_output_all)
        accuracy = metrics.accuracy_score(targets_all, new_output_all)
        print("accuracy=", accuracy)
        confusion_matrix = metrics.confusion_matrix(targets_all, new_output_all)
        print("confusion_matrix=", confusion_matrix)
        print(metrics.classification_report(targets_all, new_output_all))
        print("roc-auc score=", metrics.roc_auc_score(targets_all, outputs_all))
        res_dict = {
            'label': targets_all,
            'predict':new_output_all,
        }
        print(res_dict)
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print(f"write to {args.result_csv} succeed ")


        # # confusion_matrix
        con_mat_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print(con_mat_norm)
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        # # === plot ===
        plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_norm, annot=True, cmap='Blues', annot_kws={"fontsize":12})

        plt.ylim(0, 2)
        plt.xlabel('Predicted labels', fontsize=12)
        plt.ylabel('True labels', fontsize=12)
        plt.savefig(args.Confusion_Matrix_path)
        # # AUC
        fpr, tpr, thresholds = metrics.roc_curve(targets_all, outputs_all)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(args.ROC_path)

    print("Done Predicting")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Smiles Given an input image')
    parser.add_argument('--data_dir', default='your file where you store the test.hdf5 and test_C.hdf5 files', type=str,
                        help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--batch_size', default=24, type=int, help='Size of sampled batch')
    parser.add_argument('--output', type=str, default='output.txt',
                        help='file name to produce model predictions for each image.')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture',
                        choices=['RESNET101'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda_device', default='cuda:1', type=str, help='cuda device to use. aka gpu')
    parser.add_argument('--model_path', default='your model path', type=str, help='model path')
    parser.add_argument('--result_csv', default='./result.csv')
    parser.add_argument('--Confusion_Matrix_path', default='./Confusion_Matrix.png')
    parser.add_argument('--threshold', default=0.5, type=float, help='Size of sampled batch')
    parser.add_argument('--ROC_path', default='./roc.png')
    args = parser.parse_args()
    main(args)