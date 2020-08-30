import os
import pandas as pd
import numpy as np
import torch as th

from torch.utils.data import Dataset
from scipy.io import wavfile
from enum import Enum


# STANDARD return a speaker samples and a label (speaker id)
# ENCODER return 3 speakers samples (anchor, positive and negative) and two label (speaker id)
class ReturnType(Enum):
    STANDARD = 1
    ENCODER = 2


class IdentificationDataset(Dataset):
    
    def __init__(self, path, nb, train, encode=True, transform=None, full_sample=False, lenght=1, return_type=ReturnType.STANDARD, cross_id=None):
        if cross_id != None:
            phases = [1, 2, 3]
            file = 'iden_split_cross_' + str(cross_id) + '.txt'
        elif encode == True:
            phases = [1, 2, 3]
            if train:
                file = 'iden_split_' + str(nb) + '.txt'
            else:
                file = 'iden_split_test_' + str(nb) + '.txt'
        else:
            file = 'iden_split_test_' + str(nb) + '.txt'
            if train:
                phases = [1]
            else:
                phases = [2, 3]
            
 
        iden_split_path = os.path.join(path, file)
        # From file to table
        split = pd.read_table(iden_split_path, sep=' ', header=None, names=['phase', 'path'])
        mask = split['phase'].isin(phases)
        self.type = return_type
        # Get vector of path where mask is True
        self.dataset = split['path'][mask].reset_index(drop=True)
        self.path = path
        self.full_sample = full_sample
        self.lenght = lenght # seconds
        print("sample lenght ", self.lenght, "seconds")
        self.transform = transform
        self.nb = nb
        # number of file by idx (that a mean)
        self.half_nb_file_by_speaker = (len(self.dataset)//nb)//2
        self.min_idx = 0
        self.max_idx = len(self.dataset)

        
    def __len__(self):
        return len(self.dataset)


    def getlabel(self, idx):
        if (idx >= self.__len__()):
            return -1
        track_path = self.dataset[idx]
        label = int(track_path.split('/')[0].replace('id1', ''))
        return label
    
     
    # max distance 20 cm  || sound 343 m/sec
    # max delay correspond to 0.58 ms
    # max delay at rate 16000 is 9 samples
    def getstart(self, samples, rate):
        # Get start
        lower_bound = 0
        upper_bound = len(samples) - self.lenght * rate
        start = np.random.randint(lower_bound, upper_bound)
        return start
        
    
    def getsample(self, idx):
        # From idx get path, note that dataset is classified by label
        track_path = self.dataset[idx]
        audio_path = os.path.join(self.path, 'audio', track_path)

        # read .wav
        rate, samples = wavfile.read(audio_path)
        
        # extract label from path like id10003/L9_sh8msGV59/00001.txt
        # subtracting 1 because PyTorch assumes that C_i in [0, 1251-1]
        label = int(track_path.split('/')[0].replace('id1', ''))
            
        if self.full_sample==False:
            # segment selection
            start = self.getstart(samples, rate)
            end = start + self.lenght * rate
            samples = samples[start:end]
                
        if self.transform:
            samples = self.transform(samples)
            
        return label, samples
        
        
    def getpositiveidx(self, idx):
        if idx+self.half_nb_file_by_speaker<self.max_idx:
            maxi = idx+self.half_nb_file_by_speaker
        else:
            maxi = self.max_idx
        if idx-self.half_nb_file_by_speaker>self.min_idx:
            mini = idx-self.half_nb_file_by_speaker
        else:
            mini = self.min_idx

        p_idx = np.random.randint(mini, maxi)
        return p_idx

        
    def getnegativeidx(self, idx):
        while True:
            n_idx = np.random.randint(self.min_idx, self.max_idx)
            if (np.abs(n_idx-idx) > self.half_nb_file_by_speaker):
                break
        return n_idx


    def get_item_list_for_one_label(self, asked_label, number):
        # Get first element
        while True:            
            idx = np.random.randint(self.min_idx, self.max_idx)
            label = self.getlabel(idx)
            if  label == asked_label:
                label, sample = self.getsample(idx)
                break
        samples = th.FloatTensor(number, sample.size(0))
        samples[0] = sample
        # Get other elements
        for index in range(1, number):
            idx = self.getpositiveidx(idx)
            label = self.getlabel(idx)
            if  label == asked_label:
                label, samples[index] = self.getsample(idx)

        return samples


    def get_speakers_list(self, number):
        speakers_list = []
        idx = 0
        for j in range(self.nb):
            label = self.getlabel(idx)
            samples = self.get_item_list_for_one_label(label, number)
            speakers_list.append((label, samples))
            while label == self.getlabel(idx):
                idx += 1
        return speakers_list
        
        
    def get_item(self):
        idx = np.random.randint(self.min_idx, self.max_idx)
        label, sample = self.getsample(idx)
        return label, sample
        
    
    # Get item return an anchor, a positive and a negative item
    def __getitem__(self, idx):
        # If we are in train phase we return anchor, positive and negative
        if self.type == ReturnType.ENCODER:
            # Get hanchor
            label, sample = self.getsample(idx)
            # Get positive
            while True:
                p_idx = self.getpositiveidx(idx)
                p_label, p_sample = self.getsample(p_idx)
                if (p_label == label):
                    break

            # Get negative
            while True:
                n_idx = self.getnegativeidx(idx)
                n_label, n_sample = self.getsample(n_idx)
                if (n_label != label):
                    break

            return label, sample, p_sample, n_label, n_sample
            
        elif self.type == ReturnType.STANDARD:
            # Get sample
            label, sample = self.getsample(idx)
            return label, sample
            