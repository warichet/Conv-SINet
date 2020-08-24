# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:37:44 2020

@author: wariche1
"""
import torch as th
import torch.nn as nn
import operator


# This class corresponds to a specif place in the conference room
class Place(nn.Module):
    
    def __init__(self, id, maxsize=4):
        super(Place, self).__init__() 
        # Place id. 
        # In final version that correspond to a specific delay with a margin of error  
        self.id = id
        # Max temporal windows size in sample nb
        self.size = maxsize
        self.distances = {}


    def getid(self):
        return self.id
     
        
    # Add a distance in distance list. 
    # maxsize correspond to the maximum list size 
    def adddistance(self, label, distance):
        # Get distances list for label
        if self.distances.get(label) == None:
            self.distances[label] = []
        distances = self.distances[label]
        # Add distance in list
        if len(distances) >= self.size:
            distances.pop(0)
        distances.append(distance)
        
        
    def forward(self):
        # Calculate mean and min
        best_mean = None
        best_minimum = None
        for item in self.distances.items():
            mean = sum(item[1])
            minimum = min(item[1])
            if best_mean==None or mean<best_mean:
                mean_label = item[0]
                best_mean = mean
            if best_minimum==None or minimum<best_minimum:
                min_label = item[0]
                best_minimum = minimum
        
        return mean_label, min_label
        
        