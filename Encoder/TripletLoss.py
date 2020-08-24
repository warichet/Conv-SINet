import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):

    def __init__(self, margin=0.5, pairwise=True):
        super(TripletLoss, self).__init__()
        self.pairwise = pairwise
        if pairwise == True:
            self.dist = nn.PairwiseDistance()
            self.margin = 1.3
        else:
            self.dist = nn.CosineSimilarity(dim=1)
            self.margin = 0.1


    def distance(self, x0, x1):
        #if self.pairwise == False:
            #x0 = F.normalize(x0, dim=1, p=2)
            #x1 = F.normalize(x1, dim=1, p=2)
        return self.dist(x0, x1)


    def getResults(self, a, p, n):
        dp = self.distance(a, p)
        dn = self.distance(a, n)
        if self.pairwise== True:
            # We need a big distance for negative case and a short for positive
            return dp < dn
        else:
            # We need a big similarity for positive case and a short for negative
            return dn < dp
            

    def forward(self, a, p, n):
        dp = self.distance(a, p)
        dn = self.distance(a, n) 
        if self.pairwise== True:
            losses = F.relu(dp - dn + self.margin)
        else:
            losses = dn - dp + self.margin
        return losses.mean()
        

