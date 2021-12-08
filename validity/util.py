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


class EarlyStopping:
    def __init__(self, vars_to_save=None, save_path=None, patience=20, threshold=0.005):
        self.save_path = save_path
        self.vars_to_save = vars_to_save
        self.patience = patience
        self.threshold = threshold

        self.best_loss = None
        self.best_step = None
        self.loss_offset = 0.  # For if loss becomes negative
        self.step = 0

    def __call__(self, loss):
        # Returns true if should stop
        if self.best_loss is None:
            self.best_loss = loss
            if loss < 0.:
                self.loss_offset = 1. - loss
            self.best_step = 0
            return False

        self.step += 1

        adj_loss = loss + self.loss_offset
        adj_best_loss = self.best_loss + self.loss_offset
        if adj_loss < adj_best_loss * (1 - self.threshold):
            print(f'New best loss: {loss:0.3f}')
            self.best_loss = loss
            if loss < 0.:
                self.loss_offset = 1. - loss
            self.best_step = self.step
            if self.vars_to_save:
                torch.save(self.vars_to_save, self.save_path)
            return False

        return self.step - self.best_step >= self.patience
