# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:44:59 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn


class Conv1DBlock(nn.Module):

    def __init__(self,
                 in_channels=64,
                 out_channels=128,
                 kernel_size=5,
                 stride=1,
                 max_pool=True,
                 inception=False
                 ):       
        super(Conv1DBlock, self).__init__()  
        self.max_pool = max_pool
        self.inception = inception
        if inception:
            self.conv1 = nn.Conv1d(in_channels,
                                  out_channels//4,
                                  1,
                                  stride=1,
                                  groups=1)
            self.conv2 = nn.Conv1d(in_channels,
                                  3*out_channels//4,
                                  kernel_size,
                                  padding=kernel_size//2,
                                  stride=1,
                                  groups=1)
        else:
            self.conv = nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=2,
                                  groups=1)
            
        # Add some non-linearity
        self.acti = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        if max_pool:
            self.maxpool = nn.MaxPool1d(kernel_size=kernel_size,
                                        stride=stride//2)

        
    def forward(self, x):
        if self.inception:
            y1 = self.conv1(x)
            y2 = self.conv2(x)
            y = th.cat((y1, y2), 1)
        else:
            y = self.conv(x)
        y = self.acti(y)
        y = self.bn(y)
        if self.max_pool:
            y = self.maxpool(y)
        return y