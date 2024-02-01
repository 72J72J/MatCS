from rdkit import Chem
import networkx as nx
from utils import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import os
from sklearn import metrics
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from tqdm import tqdm
from GATencoderattentionxin import Resnet101Encoder
from utils import CaptionDataset, clip_gradient, adjust_learning_rate, AverageMeter, accuracy, set_seed
import pandas as pd
import sys, os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Preprocess training data')
    parser.add_argument('--img_size', type=int, default=256, help='Size of image X and Y dimensions')
    parser.add_argument('--label_filename', type=str, default='labels.smi',
                        help='name of labels file in case dataset is enourmous')
    parser.add_argument('--data_dir_h', default='/data/ydai/tzj/', type=str,
                        help='directory of data to be processed. Expect a labels1.smi file and associated images')
    # parser.add_argument('--data_dir_C', default=r'C:\Users\big_boss\Desktop\小论文\数据部分\数据集\training_C', type=str,
    #                     help='directory of data to be processed. Expect a labels1.smi file and associated images')
    # parser.add_argument('--data_dir', default=r'C:\Users\big_boss\Desktop\小论文\数据部分\数据集\training_C', type=str,
    #                     help='directory of data to be processed. Expect a labels.smi file and associated images')
    # parser.add_argument('--val_data_dir_C', default='/data2/ydai/GCN/data/validation_C', type=str,
    #                     help='directory of data to be processed. Expect a labels.smi file and associated images')
    # parser.add_argument('--val_data_dir', default='/data2/ydai/GCN/data/validation_H', type=str,
    #                     help='directory of data to be processed. Expect a labels.smi file and associated images')
    # parser.add_argument('--data_split', default='train', type=str,
    #                     help='name of the portion of data being processed. Typical names are training, validation, and evaluation.')
    # parser.add_argument('--config_output_name', default='dataset_img2smi.pkl', type=str,
    #                     help='name of json file to store processable metadata')
    parser.add_argument('--input_csv', default='./train.csv', type=str,
                        help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--input_csv_val', default='./validation.csv', type=str,
                        help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--output_pt', default='./processed/train.pt', type=str,
                        help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--output', default='NMR_train', type=str, help='output folder path.')
    # parser.add_argument('--process_img', action='store_true', default=False, help='create image files')
    parser.add_argument('--epochs', default=10, type=int, help='Train epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=2048, type=int, help='Size of sampled batch')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture',
                        choices=['RESNET101'])
    parser.add_argument('--cuda', default = True, help='use CUDA')
    parser.add_argument('--model_path', default='./', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='encoder learning rate if fine tuning')
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device to use. aka gpu')
    args = parser.parse_args()
    return args

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
             'encoder_optimizer': encoder_optimizer,
             'model_state_dict': encoder.state_dict(),
             'optimizer_state_dict': encoder_optimizer.state_dict()}
    filename = model_path + 'checkpoint_' + str(epoch)
    filename1 = model_path + 'bestcheck_' + str(epoch)
    torch.save(state,filename)
    if is_best:
        torch.save(state, filename1)

def main(args):
    train_smiles = []
    df = pd.read_csv('./' + 'train' + '.csv')
    train_smiles = list(df['smiles'])
    df = pd.read_csv('./' + 'validation' + '.csv')
    val_smiles = list(df['smiles'])

    train_smile_graph = {}
    val_smile_graph = {}
    for smile in train_smiles:
        g = smile_to_graph(smile)
        train_smile_graph[smile] = g
    for smile in val_smiles:
        g = smile_to_graph(smile)
        val_smile_graph[smile] = g

    processed_data_file_train =  args.output_pt

    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv(args.input_csv)
        train_drugs,  labels = list(df['smiles']),  list(df['label'])
        train_drugs,  labels = np.asarray(train_drugs), np.asarray(labels)
        df = pd.read_csv(args.input_csv_val)
        val_drugs, val_labels = list(df['smiles']), list(df['label'])
        val_drugs, val_labels = np.asarray(val_drugs), np.asarray(val_labels)

        dataset_train = 'NMR_train'
        dataset_val = 'NMR_validation'
        # make data PyTorch Geometric ready
        print('preparing ', 'davis_train.pt in pytorch format!')
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_data = TestbedDataset(args.data_dir_h, 'training_H','training_C' , transforms.Compose([normalize]), root='./', dataset= dataset_train + '_data', xd=train_drugs, y=labels,
                                    smile_graph=train_smile_graph)

        val_data = TestbedDataset(args.data_dir_h, 'validation_H','validation_C' , transforms.Compose([normalize]) , root='./', dataset= dataset_val + '_data', xd=val_drugs, y=val_labels,
                                    smile_graph=val_smile_graph)

        print(processed_data_file_train,  ' have been created')
    else:
        print('e')

    print("Creating JSON")
    best_acc = 0

    set_seed(args.seed)
    # Load Checkpoint if exists
    start_epoch = 0
    if args.load:
        try:
            print("************************")
            print("Loading models: {}".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            print("Models Loaded")
        except Exception as e:
            print(e)
            print("Models couldn't be loaded. Aborting")
            exit(0)
    else:
        if args.encoder_type == 'RESNET101':
            encoder = Resnet101Encoder()
        else:
            print("No other encoders implemented yet.")
            exit(0)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr)

        print("Models Loaded")

    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
    else:
        device = 'cpu'
    encoder = encoder.to(device)

    criterion = torch.nn.BCELoss().to(device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Custom dataloaders
    print("Loading Datasets")

    print("There are {} training data examples".format(len(train_data)))
    # print("There are {} validation data examples".format(len(val_data)))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, drop_last=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.2, patience=5,
                                                           verbose=False)

    print("Traing model")
    Loss_list = []
    Accuracy_list = []
    
    for epoch in range(start_epoch, args.epochs):
        print("Starting epoch {}".format(epoch))
        train_loss, train_acc = train(args, epoch, encoder, train_loader, encoder_optimizer, device, criterion)
        print('train_loss', train_loss)
        print('train_acc', train_acc)

        val_loss, val_acc = validate(epoch, encoder, val_loader, encoder_optimizer, device, criterion)
        scheduler.step(val_loss)

        print(f'train_loss:{train_loss}\t val_loss:{val_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')

        Loss_list.append(train_loss)
        Accuracy_list.append(train_acc)
        #
        with open(r"./train_loss.txt", 'a') as f:
            f.write(
                'epoch {e}, Loss {loss} , Accuracy {acc} \n'.format(
                    loss=train_loss, acc=train_acc, e=epoch))
            f.close()
        with open(r"./val_loss.txt", 'a') as f:
            f.write(
                'epoch {e}, Loss {loss} , Accuracy {acc} \n'.format(
                    loss=val_loss, acc=val_acc, e=epoch))
            f.close()

        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(args.model_path, epoch, encoder, encoder_optimizer, is_best, best_acc)
    # draw_loss_and_accuracy(Loss_list, Accuracy_list, args.epochs)


def train(args, epoch, encoder, train_loader, encoder_optimizer, device, criterion):
    print(epoch)
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    encoder.train()  # train mode (dropout and batchnorm is used)
    losses = AverageMeter()  # loss (per word decoded)
    train_acc = AverageMeter()  # top accuracy
    i = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        print(len(data))
        targets = data.y.view(-1, 1).float().to(device)
        encoder_optimizer.zero_grad()
        outputs = encoder(data)
        loss = criterion(outputs[0], targets.float())
        loss.backward()
        encoder_optimizer.step()
        # # measure accuracy and record loss
        acc = accuracy(outputs[0].data, targets.data, 1)
        losses.update(loss.item(), targets.size(0))
        train_acc.update(acc.item(), targets.size(0))
        print("acc", train_acc.avg)
        if i % 100 == 0:
            with open(r"./train_loss_batch.txt", 'a') as f:
                f.write(
                    'epoch {e}, batch {batch}, Loss {loss.avg} , Accuracy {acc.avg} \n'.format(
                        loss=losses, acc=train_acc, e=epoch, batch=i))

        i += 1

    return losses.avg, train_acc.avg

def validate(epoch, encoder, loader, encoder_optimizer, device, criterion):
    encoder.eval()
    losses = AverageMeter()  # loss (per word decoded)
    val_acc = AverageMeter()
    outputs_all = np.array([], dtype=int)
    targets_all = np.array([], dtype=int)
    i = 0

    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)

            targets = data.y.view(-1, 1).float().to(device)
            # Forward pass
            outputs = encoder(data)[0].to(device)

            # Calculate loss
            loss = criterion(outputs, targets.float())
            outputs_all = np.append(outputs_all, outputs[0].data)
            targets_all = np.append(targets_all, targets.data)

            losses.update(loss.item(), targets.size(0))
            acc = accuracy(outputs.data, targets.data, 1)
            val_acc.update(acc.item(), targets.size(0))

            if i % 100 == 0:
                with open(r"./val_loss_batch.txt", 'a') as f:
                    f.write(
                        'epoch {e}, batch {batch}, Loss {loss.avg} , Accuracy {acc.avg} \n'.format(
                            loss=losses, acc=val_acc, e=epoch, batch=i))
            i += 1

    new_output_all = []
    for i in range(outputs_all.shape[0]):
        if outputs_all[i] > 0.5:
            new_output_all.append(1)
        else:
            new_output_all.append(0)
    new_output_all = np.array(new_output_all)
    val_acc = metrics.accuracy_score(targets_all, new_output_all)
    return losses.avg, val_acc

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
