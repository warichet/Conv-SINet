import numpy as np

from scipy import signal

class Normalize(object):

    def __call__(self, samples):
        # removes DC component of the signal
        samples = signal.lfilter([1, -1], [1, -0.99], samples)

        # Normalize amplitude
        maximum = np.amax(samples)
        if 0.1 < maximum:
            samples = samples/maximum
        else:
            print("Error in Normalize", maximum)
        return samples