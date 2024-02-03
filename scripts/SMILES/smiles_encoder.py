import torch
from torch import nn
import torchvision


class Resnet101Encoder(nn.Module):
    """
    Resnet101 based encoder Encoder.
    """
    def __init__(self, encoded_image_size=14):
        super(Resnet101Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # pretrained ImageNet ResNet-101
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.channel_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.channel_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.embedding = nn.Sequential(
            nn.Embedding(200,256)
        )
        self.smiles =nn.Sequential(

            nn.Conv1d(256 ,32, 3,),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, 3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.layer1 = torch.nn.Sequential(
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, 96),
        )
        self.projection1 = torch.nn.Sequential(
            nn.Linear(384, 96)
        )

        self.out = torch.nn.Sequential(
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, image,  image_C, smiles):
        smiles = torch.squeeze(smiles, dim=1)
        smiles = self.embedding(smiles)
        smiles =smiles.permute(0, 2, 1)
        smiles = self.smiles(smiles)
        smiles = torch.squeeze(smiles, dim=2)

        img1 = self.resnet(image)
        img2 = self.resnet(image_C)

        # param.requires_grad = False
        image = torch.cat((img1, img2), dim=1)
        X = torch.cat((image, smiles), dim=1)
        print("#####################")
        x1 = self.layer1(X)
        x1 += self.projection1(X)
        out = self.out(x1)
        return out