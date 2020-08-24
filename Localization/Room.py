# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 06:15:05 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import numpy as np
import pyroomacoustics as pra
import random


class Room(nn.Module):
    
    def __init__(self):
        super(Room, self).__init__()
        # Create a ~4 by ~6 metres shoe box room
        rx = random.uniform(3.8, 4.2)
        ry = random.uniform(6.8, 7.2)
        self.room = pra.ShoeBox([rx, ry], fs=16000)
        # Create 2 microphones
        # 20 cm between
        self.x = random.uniform(0.5, 3.5)
        self.my = random.uniform(0.5, 1.5)
        R = np.c_[
            [self.x-0.1, self.my],  # mic 1
            [self.x+0.1, self.my],  # mic 2
            ]
        self.room.add_microphone_array(pra.Beamformer(R, self.room.fs))
        self.delay = 0
        self.rate = 16000


    def getRIR(self, mic_id, src_id):
        return room.rir[mic_id][src_id]
        
        
    def addsource(self, signal):
        # Add a source somewhere in the room
        sy = random.uniform(0.5, 3)
        self.room.add_source([self.x, self.my+sy], signal=signal, delay=self.delay)
        self.delay += signal.size(0)/self.rate

 
    def plot(self):
        fig, ax = self.room.plot()


    def forward(self):     
        self.room.simulate()
        # Get mics
        mic1 = th.from_numpy(self.room.mic_array.signals[0,:]).float()
        mic2 = th.from_numpy(self.room.mic_array.signals[1,:]).float()
        
        return mic1, mic2
