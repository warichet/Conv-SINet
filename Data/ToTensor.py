import torch
import numpy as np


class ToTensor(object):
    """Convert spectogram to Tensor."""
    
    def __call__(self, sample):
        # make the ndarray to be of a proper type (was float64)
        sample = sample.astype(np.float32)
        
        return torch.from_numpy(sample)