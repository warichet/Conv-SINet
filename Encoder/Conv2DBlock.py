# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:44:59 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn

class Conv2DBlock(nn.Module):

    def __init__(self,
                 in_channels=64,
                 out_channels=128,
                 kernel_size=7,
                 conv_stride=(1,1),
                 max_stride=(1,1),
                 padding=(0,0),
                 max_pool=True,
                 inception=False
                 ):       
        super(Conv2DBlock, self).__init__()
        self.max_pool = max_pool
        self.inception = inception
        if inception:
            self.conv1 = nn.Conv2d(in_channels,
                                 out_channels//4,
                                  1,
                                  stride=conv_stride)
            self.conv2 = nn.Conv2d(in_channels,
                                  3*out_channels//4,
                                  kernel_size,
                                  stride=conv_stride,
                                  padding=(kernel_size[0]//2,kernel_size[1]//2))
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  padding=padding,
                                  stride=conv_stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # Add some non-linearity
        self.relu = nn.ReLU()
        if max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=max_stride,
                                        stride=max_stride)

        
    def forward(self, x):
        if self.inception:
            y1 = self.conv1(x)
            y2 = self.conv2(x)
            y = th.cat((y1, y2), 1)
        else:
            y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        if self.max_pool:
            y = self.maxpool(y)
        return y
