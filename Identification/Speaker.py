# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:37:44 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# This class corresponds to a specif speaker in the conference
class Speaker(nn.Module):
    
    def __init__(self, device, label, vectors, distance=nn.PairwiseDistance(), topk=4, size=20):
        super(Speaker, self).__init__()
        self.device = device
        self.minimum = -200
        self.dist = distance
        self.label = label
        self.vectors = vectors
        self.size = vectors.size(0)
        self.topk = topk
        self.stats = th.zeros(self.size, dtype=th.int32)
        self.nb = -1
        self.means = th.zeros(self.size, dtype=th.float)


    def explore(self):
        if self.minimum < -100:
            self.minimum += 5
        
        
    def stablilize(self):
        if self.minimum > -1000:
            self.minimum -= 1


    def getlabel(self):
        return self.label
 
 
    def activatestats(self):
        self.nb = 0

 
    def setstats(self, distances):
        if self.nb == -1:
            return
        sorted, _ = th.sort(distances)
        sorted = sorted.to(self.device)
        self.means = self.means.to(self.device)
        self.means += sorted
        self.nb += 1


    def dumpstats(self):
        print("Stats for ", self.label, " : ", self.means/self.nb)
        return self.means/self.nb


    def forward(self, gold_vector, gold_label=None):
        big_gold_vector = gold_vector.repeat(self.size, 1)
        distances = self.dist(self.vectors, big_gold_vector)
        topk, _ = th.topk(distances, self.topk, largest=False)
        if self.topk > 1:
            if gold_label == self.label:
                if (topk[1]//topk[0]) > 50:
                    distances[0] = distances[1]
        # set statistics        
        self.setstats(distances)
        mean = distances.mean()
        topk = topk.sum()/self.topk
        minimum = distances.min()
        # If gold label != -1
        # We know the gold_label, that is the training phase of the second part
        # User under phone
        # Replace the worst vector by the golden vector
        # By definition golden vector is far from worst vector
        if gold_label == self.label:
            maximum = distances.max()
            ratio = int(min(10, (maximum//minimum)))
            best  = distances.argmin()
            worst = distances.argmax()
            # Keep a chance to remove
            # Explore the posible vectors, to select new reference vectors
            if self.stats[best] > self.minimum:
                self.stats[best] -= ratio
            # Don't penalize if ratio is not so big
            # Keep the elction stable, because after a while new vectors have high probability of being worse.
            if ratio >= 2:
                self.stats[worst] += ratio
                # Don't add vector too close ....
                if self.stats[worst] > 50 and minimum > 0.8:
                    self.stats[worst] = 0
                    self.vectors[worst] = gold_vector

        return mean, topk, minimum
        
        