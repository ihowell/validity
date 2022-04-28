import numpy as np
import submitit
import torch


def get_executor():
    executor = submitit.AutoExecutor(folder='logs')
    executor.update_parameters(timeout_min=7 * 24 * 60,
                               gpus_per_node=1,
                               tasks_per_node=1,
                               cpus_per_task=4,
                               slurm_partition='gpu',
                               slurm_gres='gpu',
                               slurm_mem_per_cpu='16G',
                               slurm_array_parallelism=100,
                               slurm_constraint='gpu_v100')
    return executor


def loop_gen(gen):
    while True:
        for x in gen:
            yield x


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

    def __init__(self,
                 vars_to_save=None,
                 save_path=None,
                 patience=20,
                 threshold=0.005,
                 verbose=False):
        self.save_path = save_path
        self.vars_to_save = vars_to_save
        self.patience = patience
        self.threshold = threshold
        self.verbose = verbose

        self.best_loss = None
        self.best_step = None
        self.loss_offset = 0.  # For if loss becomes negative
        self.step = 0

    def reset(self):
        self.best_loss = None
        self.best_step = None
        self.loss_offset = 0
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
            if self.verbose:
                print(f'New best loss: {loss:0.3f}')
            self.best_loss = loss
            if loss < 0.:
                self.loss_offset = 1. - loss
            self.best_step = self.step
            if self.vars_to_save:
                torch.save(self.vars_to_save, self.save_path)
            return False

        return self.step - self.best_step >= self.patience


class NPZDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        _file = np.load(self.path)
        self._data = _file['arr_0']
        self._labels = _file['arr_1']

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        data = self._data[idx]
        label = self._labels[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(data)
        return data, label


class ZipDataset(torch.utils.data.Dataset):

    def __init__(self, ds1, ds2):
        assert len(ds1) == len(ds2)
        self._ds1 = ds1
        self._ds2 = ds2

    def __len__(self):
        return len(self._ds1)

    def __getitem__(self, idx):
        return self._ds1[idx], self._ds2[idx]


class TiledDataset(torch.utils.data.Dataset):
    """
    Returns a dataset that tiles each input of the root dataset with all classes
    for each datum except for the original class. This is used for contrastive
    dataset generation.

    Assumes the underlying dataset has a `__getitem__` function that returns `(image, target)`
    where `target` is the label of the image and is in the range `[0, num_labels)`.
    """

    def __init__(self, root_dataset, num_labels):
        self._root_dataset = root_dataset
        self._num_labels = num_labels

    def __len__(self):
        return len(self._root_dataset) * (self._num_labels - 1)

    def __getitem__(self, idx):
        (image, target) = self._root_dataset[idx // (self._num_labels - 1)]
        label_offset = idx % (self._num_labels - 1)
        if label_offset >= target:
            new_target = label_offset + 1
        else:
            new_target = label_offset
        return (image, new_target)
