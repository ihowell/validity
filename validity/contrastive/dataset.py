from pathlib import Path

import fire
import torch
import numpy as np

from tqdm import tqdm

from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import ZipDataset, get_executor

from .xgems import xgems
from .cdeepex import cdeepex


def make_contrastive_dataset(contrastive_type,
                             dataset,
                             classifier_net_type,
                             classifier_weights_path,
                             generator_net_type,
                             generator_weights_path,
                             shards=20,
                             batch_size=1,
                             data_root='./datasets/',
                             cuda_idx=0,
                             seed=1,
                             **kwargs):
    assert contrastive_type in ['xgems', 'cdeepex']

    executor = get_executor()
    jobs = []
    with executor.batch():
        for i in range(shards):
            jobs.append(
                executor.submit(_make_contrastive_dataset_job, contrastive_type, dataset,
                                classifier_net_type, classifier_weights_path,
                                generator_net_type, generator_weights_path, i, shards,
                                batch_size, data_root, cuda_idx, seed, **kwargs))
    [job.result() for job in jobs]

    examples = []
    example_labels = []
    for i in range(shards):
        _file = np.load(
            f'data/tmp/{contrastive_type}_{generator_net_type}_{dataset}_{i}_{shards}.npz')
        examples.append(_file['arr_0'])
        example_labels.append(_file['arr_1'])
    examples = np.concatenate(examples)
    example_labels = np.concatenate(example_labels)

    Path('data').mkdir(exist_ok=True)
    np.savez(f'data/{contrastive_type}_{generator_net_type}_{dataset}.npz', examples,
             example_labels)


def _make_contrastive_dataset_job(contrastive_type,
                                  dataset,
                                  classifier_net_type,
                                  classifier_weights_path,
                                  generator_net_type,
                                  generator_weights_path,
                                  shard_idx,
                                  shards,
                                  batch_size=1,
                                  data_root='./datasets/',
                                  cuda_idx=0,
                                  seed=1,
                                  **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_labels = 10

    encoded_test_ds = load_encoded_ds(dataset, generator_net_type)
    _, test_ds = load_datasets(dataset)
    zip_ds = ZipDataset(test_ds, encoded_test_ds)

    n = len(zip_ds)
    shard_lower = (n * shard_idx) // shards
    shard_upper = (n * (shard_idx + 1)) // shards
    zip_ds = torch.utils.data.Subset(zip_ds, range(shard_lower, shard_upper))
    zip_loader = torch.utils.data.DataLoader(zip_ds, batch_size=batch_size, shuffle=False)

    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    generator = load_gen(generator_net_type, generator_weights_path)
    generator.eval()

    examples = []
    example_labels = []
    for (data, _), (encoded_data, _) in tqdm(zip_loader):
        for target_label in range(num_labels):
            target_label = torch.tensor(target_label).unsqueeze(0)
            print(f'{target_label=}')
            if contrastive_type == 'xgems':
                x_hat = xgems(generator,
                              classifier,
                              data,
                              target_label,
                              z_start=encoded_data,
                              **kwargs)
            elif contrastive_type == 'cdeepex':
                x_hat = cdeepex(generator,
                                classifier,
                                data,
                                target_label,
                                num_labels,
                                z_start=encoded_data,
                                **kwargs)

            examples.append(x_hat.cpu().detach().numpy())
            example_labels.append(target_label.numpy())

    examples = np.concatenate(examples)
    example_labels = np.concatenate(example_labels)

    Path('data/tmp').mkdir(exist_ok=True, parents=True)
    np.savez(
        f'data/tmp/{contrastive_type}_{generator_net_type}_{dataset}_{shard_idx}_{shards}.npz',
        examples, example_labels)


if __name__ == '__main__':
    fire.Fire()
