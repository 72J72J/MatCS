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
from smiles_encoder import Resnet101Encoder
from smiles_utils import CaptionDataset, AverageMeter, accuracy, set_seed


def save_checkpoint(model_path, epoch, encoder, encoder_optimizer, is_best, best_acc):
    print("Saving model")

    state = {'epoch': epoch,
             'best_acc': best_acc,
             'encoder': encoder,
             'encoder_optimizer': encoder_optimizer,}
    filename1 = model_path + 'bestcheck_' + str(epoch)

    if is_best:
        torch.save(state, filename1)

best_acc = 0
def main(args):
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
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
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

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_dataset = CaptionDataset(args.data_dir, 'training','training_C' ,transforms.Compose([normalize]))
    print("There are {} training data examples".format(len(train_dataset)))
    val_dataset = CaptionDataset(args.data_dir, 'validation','validation_C', transforms.Compose([normalize]))
    print("There are {} validation data examples".format(len(val_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,
                                              pin_memory=True)
    print("Datasets loaded")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    print("Traing model")
    Loss_list = []
    Accuracy_list = []
    best_acc = 0
    for epoch in range(start_epoch, args.epochs):
        print("Starting epoch {}".format(epoch))
        train_loss,train_acc = train(args, epoch, encoder, train_loader, encoder_optimizer, device, criterion)
        val_loss,val_acc = validate(epoch, encoder, val_loader, encoder_optimizer, device,criterion)
        scheduler.step(val_loss)

        print(f'train_loss:{train_loss}\t val_loss:{val_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')

        Loss_list.append(train_loss)
        Accuracy_list.append(train_acc)
        #
        with open(r"./train_loss.txt", 'a') as f:
            print("save loss")
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

        '''save model'''
        save_checkpoint(args.model_path, epoch, encoder, encoder_optimizer,is_best,best_acc)


def train(args, epoch, encoder,  loader,  encoder_optimizer, device, criterion):
    print(epoch)
    encoder.train()  # train mode (dropout and batchnorm is used)
    losses = AverageMeter()  # loss (per word decoded)
    train_acc = AverageMeter()  # top accuracy
    i = 0
    for data in tqdm(loader):
        imgs = data[0]
        targets = data[2]
        imgsC = data[3]
        smiles = data[4]

        imgs = imgs.to(device)
        imgsC = imgsC.to(device)
        targets = targets.to(device)
        smiles = smiles.to(device)

        encoder_optimizer.zero_grad()
        outputs= encoder(imgs, imgsC,smiles)
        loss = criterion(outputs,targets.float())
        loss.backward()
        encoder_optimizer.step()

        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data,1)
        losses.update(loss.item(), imgs.size(0))
        train_acc.update(acc.item(), imgs.size(0))
        print("acc", train_acc.avg)
        if i % 100 == 0:
            with open(r"./train_loss_batch.txt", 'a') as f:
                f.write(
                    'epoch {e}, batch {batch}, Loss {loss.avg} , Accuracy {acc.avg} \n'.format(
                        loss=losses, acc=train_acc, e=epoch,batch= i))

        i += 1

    return losses.avg,train_acc.avg


def validate(epoch, encoder,loader, encoder_optimizer, device, criterion):
    encoder.eval()
    losses = AverageMeter()  # loss (per word decoded)
    val_acc = AverageMeter()
    outputs_all = np.array([], dtype=int)
    targets_all = np.array([], dtype=int)
    i = 0

    with torch.no_grad():
        for data in tqdm(loader):
            imgs = data[0]
            targets = data[2]
            imgsC = data[3]
            imgs = imgs.to(device)

            targets = targets.to(device)
            imgsC = imgsC.to(device)
            smiles = data[4]
            smiles = smiles.to(device)
            # Forward pass
            outputs = encoder(imgs,imgsC,smiles).to(device)

            # Calculate loss
            loss = criterion(outputs,targets.float())
            outputs_all = np.append(outputs_all, outputs.data)
            targets_all = np.append(targets_all, targets.data)

            losses.update(loss.item(), imgs.size(0))
            acc = accuracy(outputs.data, targets.data, 1)
            val_acc.update(acc.item(), imgs.size(0))

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
    parser = argparse.ArgumentParser(description='Train molecule captioning model')
    parser.add_argument('--data_dir', default='your file where you store the .hdf5 files', type=str,
                        help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--epochs', default=200, type=int, help='Train epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=4, type=int, help='Size of sampled batch')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture',
                        choices=['RESNET101'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--model_path', default='your model path', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='encoder learning rate if fine tuning')
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device to use. aka gpu')
    args = parser.parse_args()
    main(args)
