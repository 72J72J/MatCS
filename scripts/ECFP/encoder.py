import torch
from torch import nn
import torchvision

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Resnet101Encoder(nn.Module):
    """
    Resnet101 based encoder Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Resnet101Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.resnet = torchvision.models.resnet101(pretrained=True)
        # pretrained ImageNet ResNet-101
        
        self.channel_in = self.resnet.fc.in_features
        modules = list(self.resnet.children())[:-2]  # Remove linear and pool layers (since we're not doing classification)
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.resnet_fc = nn.Sequential(

            nn.Linear(self.channel_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        

        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()

        self.morgen = nn.Sequential(

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
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


    def forward(self, image, morgenfp,image_C):

        img1 = self.resnet(image)
        img2 = self.resnet(image_C)
        
        img1 = self.avg_pool(img1)
        img2 = self.avg_pool(img2)

        img1 = img1.reshape(img1.size(0), -1)
        img2 = img2.reshape(img2.size(0), -1)

        img1 = self.resnet_fc(img1)
        img2 = self.resnet_fc(img2)

        image = torch.cat((img1, img2), dim=1)

        morgenfp = morgenfp.float()
        morgenfp = self.morgen(morgenfp)

        print(morgenfp.shape)
        X = torch.cat((image, morgenfp), dim=1)
        x1 = self.layer1(X)
        x1 += self.projection1(X)
        out = self.out(x1)

        return out