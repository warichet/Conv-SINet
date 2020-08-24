# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 06:15:05 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import numpy as np
import random

from Conv1DBlock import Conv1DBlock
from TripletLoss import TripletLoss


class LocateEncoder(nn.Module):
    
    def __init__(self, final_size=32):
        super(LocateEncoder, self).__init__()
        # Encoder without stride
        self.conv1 = Conv1DBlock(in_channels=1,
                            out_channels=final_size//4,
                            kernel_size=5,
                            stride=1,
                            max_pool=False,
                            inception=False)
        self.conv2 = Conv1DBlock(in_channels=final_size//4,
                            out_channels=final_size,
                            kernel_size=5,
                            stride=1,
                            max_pool=False,
                            inception=False)
        self.conv3 = Conv1DBlock(in_channels=final_size,
                            out_channels=final_size,
                            kernel_size=5,
                            stride=1,
                            max_pool=False,
                            inception=True)
        self.conv4 = Conv1DBlock(in_channels=final_size,
                            out_channels=final_size,
                            kernel_size=5,
                            stride=1,
                            max_pool=False,
                            inception=True)
        self.conv5 = Conv1DBlock(in_channels=final_size,
                            out_channels=final_size,
                            kernel_size=5,
                            stride=1,
                            max_pool=False,
                            inception=True)
        self.loss = TripletLoss(0.55, pairwise=True)
        self.encoder_path = '/content/drive/My Drive/Siamese/Localization/encoder.pt'


    def store_encoder(self):
        net = th.save(self.state_dict(), self.encoder_path)
        return net
        

    # Select a random number in an interval of exclude-10 <-> exclude+10
    # Remove exclude from this intervale.
    def random_exclude(self, size, exclude):
        minimum = max(0, exclude-10)
        maximum = min(size-1, exclude+10)
        randInt = np.random.randint(minimum, maximum)
        if randInt == exclude:
            return self.random_exclude(size, exclude)  
        else:
            return randInt 


    # We have sample from two mics x1 and x2
    def forward(self, x1, x2):
        # Encode the 2 mics
        x1 = x1[None, None, :]
        y1 = self.conv1(x1)
        y1 = self.conv2(y1)
        y1 = self.conv3(y1)
        y1 = self.conv4(y1)
        y1 = self.conv5(y1)
        x2 = x2[None, None, :]
        y2 = self.conv1(x2)
        y2 = self.conv2(y2)
        y2 = self.conv3(y2)
        y2 = self.conv4(y2)
        y2 = self.conv5(y2)

        # Get loss
        loss = 0
        y_len = y1.size(2)
        
        results = None
        nb = 0
        for tp in range(0, y_len, 100):
            nb += 1
            a = y1[:, :, tp]
            p = y2[:, :, tp]
            # Select a negative
            tn = self.random_exclude(y_len, tp)
            n = y2[:, :, tn]
            # Get loss
            loss += self.loss(a, p, n)
            # loss.getResults() return an array of mini batch size.
            # With 0 value corresponding to an error, and 1 value for right case.
            if results == None:
                results = self.loss.getResults(a, p, n).float()
            else:
                results += self.loss.getResults(a, p, n).float()
        
        # results/nb corresponds to the accuracy
        return results/nb, loss
