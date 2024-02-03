from PIL import Image
import cv2
import pickle
from rdkit import Chem
import networkx as nx
from utils import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import pandas as pd
from sklearn import metrics
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from tqdm import tqdm
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns


def create_input_files(args, H_dataset_path, C_dataset_path, config_output_name, img_size,
                       dataset_size=4000000):
    '''
    Creates input files for using with models.
    :param dataset_path: the path to the data to be processed
    '''
    data_H = pickle.load(open(os.path.join(H_dataset_path, config_output_name), 'rb'))
    data_C = pickle.load(open(os.path.join(C_dataset_path, config_output_name), 'rb'))
    images_H = {}
    images_C = {}
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([normalize])

    if True:
        print("Processing images")
        for i, cur_data in enumerate(tqdm(data_H['images'])):
            print(i)

            if i > dataset_size:
                break
            path = os.path.join(cur_data['filepath'], cur_data['filename'])
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.array(Image.fromarray(img).resize((img_size, img_size)))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, img_size, img_size)
            assert np.max(img) <= 255
            # Save image to HDF5 file
            img = torch.FloatTensor(img / 255.)
            img = transform(img)
            img = img.numpy()
            images_H[i] = img

        for i, cur_data in enumerate(tqdm(data_C['images'])):
            print(i)

            if i > dataset_size:
                break
            path = os.path.join(cur_data['filepath'], cur_data['filename'])
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.array(Image.fromarray(img).resize((img_size, img_size)))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, img_size, img_size)
            assert np.max(img) <= 255

            img = torch.FloatTensor(img / 255.)
            img = transform(img)
            img = img.numpy()
            images_C[i] = img

    else:
        print("Just processing captions")

    return images_H, images_C


def create_tokenized_smiles_json(H_data_dir, C_data_dir, split, config_output_name, label_filename):
    data = {"images": []}

    with open(os.path.join(H_data_dir, label_filename), "r") as f:
        for i, l in enumerate(tqdm(f)):
            try:
                smiles, idx, label = l.strip().split("\t")
            except:
                pass

            current_sample = {"filepath": H_data_dir, "filename": "{}".format(idx), "imgid": 0, "split": split,
                              "sentences": [{"raw": smiles,
                                             "label": label}]}
            data["images"].append(current_sample)
    pickle.dump(data, open(os.path.join(H_data_dir, config_output_name), 'wb'))
    del data
    data_C = {"images": []}
    with open(os.path.join(C_data_dir, label_filename), "r") as f:
        for i, l in enumerate(tqdm(f)):
            try:
                smiles, idx, label = l.strip().split("\t")

            except:
                pass

            current_sample = {"filepath": C_data_dir, "filename": "{}".format(idx), "imgid": 0, "split": split,
                              "sentences": [{"raw": smiles,
                                             "label": label}]}

            data_C["images"].append(current_sample)
    pickle.dump(data_C, open(os.path.join(C_data_dir, config_output_name), 'wb'))
    del data_C


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


def save_checkpoint(model_path, epoch, encoder, encoder_optimizer, is_best, best_acc):
    print("Saving model")

    state = {'epoch': epoch,
             'best_acc': best_acc,
             'encoder': encoder,
             'encoder_optimizer': encoder_optimizer, }
    filename1 = model_path + 'bestcheck_' + str(epoch)

    if is_best:
        torch.save(state, filename1)


best_acc = 0


def main(args):
    df = pd.read_csv('./' + 'test' + '.csv')
    val_smiles = list(df['smiles'])

    print("test_smiles:", len(val_smiles))

    val_smile_graph = {}
    for smile in val_smiles:
        g = smile_to_graph(smile)
        val_smile_graph[smile] = g

    processed_data_file_train = args.output_pt

    if ((not os.path.isfile(processed_data_file_train))):

        df = pd.read_csv(args.input_csv)
        val_drugs, val_labels = list(df['smiles']), list(df['label'])

        val_drugs, val_labels = np.asarray(val_drugs), np.asarray(val_labels)

        # dataset_train = 'NMR_train'
        dataset_val = 'NMR_validation'
        # make data PyTorch Geometric ready
        print('preparing ', 'davis_train.pt in pytorch format!')
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        val_data = TestbedDataset(args.data_dir_h, 'test', 'test_C', transforms.Compose([normalize]),
                                  root='./', dataset=dataset_val + '_data', xd=val_drugs, y=val_labels,
                                  smile_graph=val_smile_graph)

        print(processed_data_file_train, ' have been created')
    else:
        print(processed_data_file_train, ' are already created')

    print("Creating JSON")

    set_seed(args.seed)

    try:
        print("Loading models: {}".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        encoder = checkpoint['encoder']

        print("Models Loaded")
    except:
        print("Models couldn't be loaded. Aborting")
        exit(0)

    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
    else:
        device = 'cpu'
    encoder = encoder.to(device)
    encoder.eval()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Custom dataloaders
    print("Loading Datasets")
    print("There are {} test data examples".format(len(val_data)))

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
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
            outputs = encoder(data).to(device)

            y_true.extend(targets.detach().tolist())
            print(outputs)
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
            'predict': new_output_all,
            'predict2': outputs_all,
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
        sns.heatmap(con_mat_norm, annot=True, cmap='Blues', annot_kws={"fontsize": 20})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 2)
        plt.xlabel('Predicted labels', fontsize=20)
        plt.ylabel('True labels', fontsize=20)
        plt.savefig(args.Confusion_Matrix_path)
        # # AUC
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
        plt.savefig(args.ROC_path)

    print("Done Predicting")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess training data')
    parser.add_argument('--img_size', type=int, default=256, help='Size of image X and Y dimensions')
    parser.add_argument('--label_filename', type=str, default='labels.smi',
                        help='name of labels file in case dataset is enourmous')
    parser.add_argument('--data_dir_h', default='your file where you store the test.hdf5 and test_C.hdf5 files', type=str,
                        help='directory of data to be processed. Expect a labels1.smi file and associated images')
    parser.add_argument('--config_output_name', default='dataset_img2smi.pkl', type=str,
                        help='name of json file to store processable metadata')
    parser.add_argument('--input_csv', default='./test.csv', type=str,
                        help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--output_pt', default='./processed/train.pt', type=str,
                        help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--output', default='NMR_train', type=str, help='output folder path.')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=8, type=int, help='Size of sampled batch')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture',
                        choices=['RESNET101'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--model_path', default='your modelpath', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--cuda_device', default='cuda:2', type=str, help='cuda device to use. aka gpu')
    parser.add_argument('--result_csv', default='result.csv')
    parser.add_argument('--Confusion_Matrix_path', default='Confusion_Matrix.png')
    parser.add_argument('--threshold', default=0.5, type=float, help='Size of sampled batch')
    parser.add_argument('--ROC_path', default='roc.png')

    args = parser.parse_args()
    main(args)