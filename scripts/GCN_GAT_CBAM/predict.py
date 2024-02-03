import os
import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from sklearn import metrics
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils import set_seed
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def main():
    ## load data
    data_dir_h = 'your file where you store the test.hdf5 and test_C.hdf5 files'
    input_csv_val = 'test.csv'
    ## output path
    result_csv = './result.csv'
    Confusion_Matrix = './confusion_matrix.png'
    ROC_path = '/ROC.png'
    ## model path
    model_path = 'your model'

    threshold = 0.5
    seed = 42
    batch_size = 64

    df = pd.read_csv(input_csv_val)
    val_drugs, val_labels = list(df['smiles']), list(df['label'])
    print("test_smiles:", len(val_drugs))
    val_smile_graph = {}
    for smile in val_drugs:
        # print(smile)
        g = smile_to_graph(smile)
        val_smile_graph[smile] = g
    val_drugs, val_labels = np.asarray(val_drugs), np.asarray(val_labels)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    val_data = TestbedDataset(data_dir_h, 'test', 'test_C', transforms.Compose([normalize]),
                                root='./', dataset='NMR_validation' + '_data', xd=val_drugs, y=val_labels,
                                smile_graph=val_smile_graph)

    set_seed(seed)
    # Load Checkpoint if exists
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models: {}".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    encoder = checkpoint['encoder']
    print("Models Loaded")

    encoder = encoder.to(device)
    encoder.eval()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Custom dataloaders
    print("Loading Datasets")
    print("There are {} test data examples".format(len(val_data)))

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    print("Predicting images")
    y_true = []
    outputs_all = np.array([], dtype=int)
    targets_all = np.array([], dtype=int)

    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data.to(device)

            targets = data.y.view(-1, 1).float().to(device)
            # Forward pass
            outputs = encoder(data)
            y_true.extend(targets.detach().tolist())
            outputs_all = np.append(outputs_all, outputs[0].cpu().numpy())
            targets_all = np.append(targets_all, targets.cpu().numpy()).tolist()

        new_output_all = []
        for i in range(outputs_all.shape[0]):
            if outputs_all[i] > threshold:
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
            'predict': new_output_all,
            'predict2': outputs_all,
        }
        print(res_dict)
        df = pd.DataFrame(res_dict)
        df.to_csv(result_csv, index=False)
        print(f"write to {result_csv} succeed ")

        con_mat_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print(con_mat_norm)
        con_mat_norm = np.around(con_mat_norm, decimals=2)

        # === plot ===
        plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_norm, annot=True, cmap='Blues', annot_kws={"fontsize": 20})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 2)
        plt.xlabel('Predicted labels', fontsize=20)
        plt.ylabel('True labels', fontsize=20)
        # plt.show()
        plt.savefig(Confusion_Matrix)
        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(targets_all, outputs_all)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(ROC_path)

    print("Done Predicting")

if __name__ == '__main__':
    main()