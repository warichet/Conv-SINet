# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:55:36 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from Time import Time
from TransFourier import TransFourier
from Speaker import Speaker
from Place import Place

# This class corresponds to a conference room
class Conference(nn.Module):

    def __init__(self,
                device,
                avg=True,
                time=True):
        super(Conference, self).__init__()
        self.device = device
        self.encoder_path = '/content/drive/My Drive/Siamese/Encoder/Saved/'
        self.speakers_path = '/content/drive/My Drive/Siamese/Identification/Saved/speakers_'
        # Select time encoder or frequency encoder            
        if time:
            self.encoder = Time(avg=avg)
            self.time = True
        else:
            self.encoder = TransFourier()
            self.time = False
        # List of speakers
        self.speakers = {}
        # List of places
        self.places = {}

    # Load the specific encoder
    def load_encoder(self, size=3):
        if self.time == False:
            encoder_path = self.encoder_path + 'encoder_frequency_'+ str(size) + '.pt'
        else:
            encoder_path = self.encoder_path + 'encoder_time_'+ str(size) + '.pt'
        self.encoder.load_state_dict(th.load(encoder_path))
        print("Encoder loaded")

            
    # Encode sample into vector
    def getvector(self, sample):
        y = self.encoder.getvector(sample)
        return y


    # Init the speakers list 
    def initspeakers(self, speakers, topk=4):
        self.pool_size = len(speakers)
        # For each speaker encode samples into vectors and store it 
        self.speakers.clear()
        for (label, samples) in speakers:
            samples = samples.to(self.device)
            vectors = self.getvector(samples)
            speaker_obj = Speaker(self.device, label, vectors, topk=topk)
            self.speakers[label] = speaker_obj


    # Init the place list
    def initplaces(self, speakers, maxsize=4):
        # For each speaker encode samples into vectors and store it 
        self.places.clear()
        for (label, samples) in speakers:
            place_obj = Place(label, maxsize=maxsize)
            self.places[label] = place_obj


    def initplaces(self, maxsize=4):
        # For each speaker encode samples into vectors and store it 
        self.places.clear()
        for label in self.speakers:
            place_obj = Place(label, maxsize=maxsize)
            self.places[label] = place_obj

    
    def forward(self, sample, gold_label=None, place_id=None):
        sample = sample.to(self.device)
        # Encode the sample
        vector = self.getvector(sample)
        # Get metrics
        best_mean = None
        best_topk = None
        best_min = None
        place = None
        
        # Select a Place
        # ToDo select the right place using sample and not gold_label
        if place_id:
            place = self.getplace(sample, place_id)
        
        for label in self.speakers:
            speaker = self.speakers[label]
            mean, topk, minimum = speaker(vector, gold_label)
            if place:
                place.adddistance(label, topk)
            else:
                if (best_mean == None) or (mean < best_mean):
                    best_mean = mean
                    mean_label = label
                if (best_topk == None) or (topk < best_topk):
                    best_topk = topk
                    topk_label = label
                    best_speaker = speaker
                if (best_min == None) or (minimum < best_min):
                    best_min = minimum
                    min_label = label
        # Case where place is used
        if place:    
            mean_label, min_label = place()
            return mean_label, min_label
        # Case where vectors are learned
        if mean_label != gold_label:
            best_speaker.explore()
        else:
            best_speaker.stablilize()
        return mean_label, topk_label, min_label


    # Store speaker list into file
    def store(self, nb_speakers , size=3, cross_id=None):
        if cross_id:
            speakers_path = self.speakers_path + str(nb_speakers) + '_' + str(size) + '_' + str(cross_id) + '.pt'
        else:
            speakers_path = self.speakers_path + str(nb_speakers) + '_' + str(size) + '.pt'
        print("Store network", speakers_path)
        th.save(self.speakers, speakers_path)


    # Load speaker list from file
    def load(self, nb_speakers, size=3, cross_id=None):
        if cross_id:
            speakers_path = self.speakers_path + str(nb_speakers) + '_' + str(size) + '_' + str(cross_id) + '.pt'
        else:
            speakers_path = self.speakers_path + str(nb_speakers) + '_' + str(size) + '.pt'
        print("Load ", speakers_path)
        self.speakers = th.load(speakers_path)


    # In the absence of localization simulate the places
    def getplace(self, sample, label):
        return  self.places[label]


    # Tools for statistics
    def activatestats(self):
        for label in self.speakers:
            speaker = self.speakers[label]
            speaker.activatestats()
    
    
    def dumpstats(self):
        nb = 0
        means = th.zeros(self.pool_size, dtype=th.float)
        means = means.to(self.device)
        for label in self.speakers:
            speaker = self.speakers[label]
            means += speaker.dumpstats()
            nb += 1
        print("MEANS : ", means/nb)
        return means/nb
