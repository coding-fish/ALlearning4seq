import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net=None):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n):
        picked_idxs = [(idx, 0) for idx in range(len(self.dataset))]
        np.random.seed(42)
        np.random.shuffle(picked_idxs)
        return picked_idxs[:n]
