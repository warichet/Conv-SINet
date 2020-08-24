# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:55:36 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import torchaudio

from scipy import signal

from Conv2DBlock import Conv2DBlock
from TripletLoss import TripletLoss


class TransFourier(nn.Module):

    def __init__(self, pairwise=True):
        super(TransFourier, self).__init__()
        self.pairwise = pairwise
        self.encoder_path = '/content/drive/My Drive/Siamese/Encoder/Saved/'
        rate = 16000
        # window width and step size
        Tw = 25 # ms
        Th = 10 # ms
        # frame duration (samples)
        Nw = int(rate * Tw * 1e-3)
        Nh = int(rate * Th * 1e-3)
        # overlapped duration (samples)
        nfft = 2 ** (Nw.bit_length()+1)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=nfft, win_length=Nw, hop_length=Nh, power=1, normalized=False)
        self.amp = torchaudio.transforms.AmplitudeToDB()
        # Start convolution
        self.block1 = Conv2DBlock(in_channels=1,
                                 out_channels=128,
                                 kernel_size=(7,7),
                                 conv_stride=(2,1),
                                 max_stride=(4,4),
                                 padding=(0,0),
                                 inception=True)
        self.block2 = Conv2DBlock(in_channels=128,
                                 out_channels=256,
                                 kernel_size=(5,5),
                                 conv_stride=(2,1),
                                 max_stride=(4,2),
                                 padding=(0,0),
                                 inception=True)
        self.block3 = Conv2DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=(3,3),
                                 conv_stride=(2,1),
                                 max_pool=False,
                                 inception=True)
        self.block4 = Conv2DBlock(in_channels=256,
                                 out_channels=256,
                                 kernel_size=(4,3),
                                 conv_stride=(2,1),
                                 max_pool=False,
                                 inception=False)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 1, stride=1, groups=1, bias=False)
        
        self.loss = TripletLoss(0.55, pairwise=pairwise)


    def getvector(self, x):
        # Modify raw data, extract Frequency components
        x = self.spectrogram(x)
        x = self.amp(x)
        x = x[:, None, :]
        # x dim = (mini batch, channels=1, frequency, time)
        # Start 2D convolution
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        # Use average pooling to force the NN to generalize.
        y = self.apool(y)
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
    
    def get_spec(self, x):
        y = self.spectrogram(x)
        y = self.amp(y)
        return y
        
        
    def store_encoder(self, lenght):
        if self.pairwise:
            encoder_path = self.encoder_path + 'encoder_frequency_' + str(lenght) + ".pt"
        else:
            encoder_path = self.encoder_path + 'encoder_frequency_'+ str(lenght) + "_cos.pt"
            
        net = th.save(self.state_dict(), encoder_path)
        return net
