from importlib_metadata import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import torchvision



def ConvBNRelu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )

def ConvBNRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class DctStem(nn.Module): # [3,3,3]   # [32,64,128]
    def __init__(self, kernel_sizes, num_channels):
        super(DctStem, self).__init__()
        self.convs = nn.Sequential(
            ConvBNRelu2d(in_channels=1,
                         out_channels=num_channels[0], # 32
                         kernel_size=kernel_sizes[0]), # (1,3)
            ConvBNRelu2d(
                in_channels=num_channels[0], # 32
                out_channels=num_channels[1], # 64
                kernel_size=kernel_sizes[1], # (1,3)
            ),
            ConvBNRelu2d(
                in_channels=num_channels[1], #  64
                out_channels=num_channels[2], # 128
                kernel_size=kernel_sizes[2], # (1,3)
            ),
            nn.MaxPool2d((1, 2)),
        )

    def forward(self, dct_img): # dct_img的格式是：[B,64,250]
        x = dct_img.unsqueeze(1) # x的格式：[B,1,64,250]
        img = self.convs(x)
        img = img.permute(0, 2, 1, 3)  # transpose是一次转变一对维度，permute可以一次转变多个维度

        return img

class DctInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channel=128,
        branch1_channels=[64],
        branch2_channels=[48, 64],
        branch3_channels=[64, 96, 96],
        branch4_channels=[32],
    ):
        super(DctInceptionBlock, self).__init__()

        self.branch1 = ConvBNRelu2d(in_channels=in_channel,
                                    out_channels=branch1_channels[0],
                                    kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch2_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch2_channels[0],
                out_channels=branch2_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch3 = nn.Sequential(
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch3_channels[0],
                         kernel_size=1),
            ConvBNRelu2d(
                in_channels=branch3_channels[0],
                out_channels=branch3_channels[1],
                kernel_size=3,
                padding=(0, 1),
            ),
            ConvBNRelu2d(
                in_channels=branch3_channels[1],
                out_channels=branch3_channels[2],
                kernel_size=3,
                padding=(0, 1),
            ),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBNRelu2d(in_channels=in_channel,
                         out_channels=branch4_channels[0],
                         kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        # y = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = out.permute(0, 2, 1, 3)
        return out


class DctCNN(nn.Module):
    def __init__(self,
                 model_dim,     # 256
                 dropout,       # 0.5
                 kernel_sizes,  # [3,3,3]
                 num_channels,  # [32,64,128]
                 in_channel=128,
                 branch1_channels=[64],
                 branch2_channels=[48, 64],
                 branch3_channels=[64, 96, 96],
                 branch4_channels=[32],
                 out_channels=64):
        super(DctCNN, self).__init__()

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = DctInceptionBlock(
            in_channel,
            branch1_channels,
            branch2_channels,
            branch3_channels,
            branch4_channels,
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = ConvBNRelu2d(branch1_channels[-1] + branch2_channels[-1] +
                                 branch3_channels[-1] + branch4_channels[-1],
                                 out_channels,
                                 kernel_size=1)

    def forward(self, dct_img):  # 输入的图片格式是[B,64,250]
        dct_f = self.stem(dct_img) # dct_f的格式：[B,64,128,122]
        x = self.InceptionBlock(dct_f) # [B,64,256,122]
        x = self.maxPool(x) # [B,64,256,1]
        x = x.permute(0, 2, 1, 3) # [B,256,64,1]
        x = self.conv(x)  # [B,64,64,1]
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)
        x = x.reshape(-1, 4096)
        return x  #  最终输出是[B,4096]