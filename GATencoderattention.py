import torch
from torch import nn
import torchvision
import torch.nn.functional as f
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

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
        # print(x.shape)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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

    def __init__(self, encoded_image_size=14, n_output=1, n_filters=32, embed_dim=128,num_features_xd=38, output_dim=128, dropout=0.0 ):
        super(Resnet101Encoder, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, self.n_output)

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
            #nn.Dropout(0.1),  # 0.2
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()

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

    def forward(self, data):


        x, edge_index, batch, image,  image_C = data.x, data.edge_index, data.batch, data.target, data.nmr_C
        print(x.shape)  # (16568,78)
        print(edge_index.shape)  # (2,36640)
        print(batch.shape)  # 16568
        print('target', image_C.shape)

        x , attn_weights = self.conv1(x, edge_index, return_attention_weights = True)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        # attention = x
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)


        print("x", x.shape)  # x torch.Size([32, 128])
        #print('img1',image)

        img1 = self.resnet(image)
        # print(img1)

        print(img1.shape)
        img2 = self.resnet(image_C)
        img1 = self.ca(img1) * img1
        img1 = self.sa(img1) * img1
        img2 = self.ca(img2) * img2
        img2 = self.sa(img2) * img2
        #
        img1 = self.avg_pool(img1)
        img2 = self.avg_pool(img2)
        # # print(img1.shape)
        img1 = img1.reshape(img1.size(0), -1)
        img2 = img2.reshape(img2.size(0), -1)
        #
        img1 = self.resnet_fc(img1)
        img2 = self.resnet_fc(img2)
        print(img1.shape)

        # param.requires_grad = False
        image = torch.cat((img1, img2), dim=1)


        # 特征融合
        X = torch.cat((image, x), dim=1)
        x1 = self.layer1(X)
        x1 += self.projection1(X)
        out = self.out(x1)
        return out , attn_weights

        #print(image)
