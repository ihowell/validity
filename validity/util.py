import torch
import numpy as np


class np_loader:
    def __init__(self, ds, label_is_ones):
        self.ds = ds
        self.label_is_ones = label_is_ones

    def __iter__(self):
        if self.label_is_ones:
            label = np.ones(64)
        else:
            label = np.zeros(64)

        for i in range(self.ds.shape[0] // 64):
            batch = self.ds[i * 64:(i + 1) * 64]
            yield torch.tensor(batch), torch.tensor(label)
