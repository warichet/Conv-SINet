# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 06:15:05 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn

from Conv1DBlock import Conv1DBlock
from TripletLoss import TripletLoss


class Time(nn.Module):
    
    def __init__(self,
                 avg=True,
                 freq=4000):
        super(Time, self).__init__()
        self.encoder_path = '/content/drive/My Drive/Siamese/Encoder/Saved/'
        # Available stride correspond to a stride that cut space in word granularity
        if avg:        
            rate = 16000
            stride = rate//freq
            self.block0 = nn.AvgPool1d(kernel_size=7, stride=stride)
            self.block1 = Conv1DBlock(in_channels=1,
                                     out_channels=64,
                                     kernel_size=9,
                                     stride=16)
        else:
            self.block0 = Conv1DBlock(in_channels=1,
                                     out_channels=16,
                                     kernel_size=9,
                                     stride=16)
            self.block1 = Conv1DBlock(in_channels=16,
                                     out_channels=128,
                                     kernel_size=9,
                                     stride=8)
                                     
        self.block2 = Conv1DBlock(in_channels=128,
                                 out_channels=256,
                                 kernel_size=9,
                                 stride=8)
        self.block3 = Conv1DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 max_pool=False,
                                 inception=True)
        self.block4 = Conv1DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 max_pool=False,
                                 inception=True)
        self.block5 = Conv1DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 max_pool=False,
                                 inception=True)
        self.block6 = Conv1DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 max_pool=False,
                                 inception=True)
        self.block7 = Conv1DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=9,
                                 max_pool=False,
                                 inception=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, 1, stride=1, groups=1, bias=False)
        
        # Distance between 2 vectors
        self.dist = nn.PairwiseDistance()
        self.loss = TripletLoss(0.55, distance=self.dist)
            

    def getvector(self, x):
        # create a tensor of size [batch size, one channel, L data]
        x = x[:, None, :]
        y = self.block0(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.avgpool(y)
        # Flatten the output
        y = y.view(y.size(0), -1)
        y = y[:, None, :]
        y = self.conv(y)
        y = y.view(y.size(0), -1)  

        return y


    def forward(self, a, p , n):
        # Get vectors
        a = self.getvector(a)
        p = self.getvector(p)
        n = self.getvector(n)

        # Get loss
        loss = self.loss(a, p, n)
        # getResult
        result = self.loss.getResults(a, p, n)
        
        return loss, result, a, p, n
    
        
    def store_encoder(self, size):
        encoder_path = self.encoder_path + 'encoder_time_'+ str(size) + '.pt'
        net = th.save(self.state_dict(), encoder_path)
        return net