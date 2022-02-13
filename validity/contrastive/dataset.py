from pathlib import Path

import fire
import torch
import numpy as np

from tqdm import tqdm

from validity.classifiers import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen, load_encoded_ds
from validity.util import ZipDataset, get_executor

from .am import am
from .cdeepex import cdeepex
from .xgems import xgems


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
                             seed=0,
                             dry_run_size=None,
                             **kwargs):
    assert contrastive_type in ['am', 'xgems', 'cdeepex']

    executor = get_executor()
    jobs = []
    with executor.batch():
        for i in range(shards):
            jobs.append(
                executor.submit(_make_contrastive_dataset_job, contrastive_type, dataset,
                                classifier_net_type, classifier_weights_path,
                                generator_net_type, generator_weights_path, i, shards,
                                batch_size, data_root, cuda_idx, seed, dry_run_size, **kwargs))
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
    save_path = get_contrastive_dataset_path(contrastive_type, dataset, classifier_net_type,
                                             generator_net_type)
    np.savez(str(save_path), examples, example_labels)


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
                                  dry_run_size=None,
                                  **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_labels = 10

    print('Loading datasets')
    _, test_ds = load_datasets(dataset)
    encoded_test_ds = load_encoded_ds(dataset, generator_net_type)

    if dry_run_size:
        test_ds = torch.utils.data.Subset(test_ds, range(dry_run_size))
        encoded_test_ds = torch.utils.data.Subset(encoded_test_ds, range(dry_run_size))

    zip_ds = ZipDataset(test_ds, encoded_test_ds)

    print('Sharding dataset')
    n = len(zip_ds)
    shard_lower = (n * shard_idx) // shards
    shard_upper = (n * (shard_idx + 1)) // shards
    zip_ds = torch.utils.data.Subset(zip_ds, range(shard_lower, shard_upper))
    zip_loader = torch.utils.data.DataLoader(zip_ds, batch_size=batch_size, shuffle=False)

    print('Loading classifier')
    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier.eval()

    print('Loading generator')
    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator.eval()

    examples = []
    example_labels = []

    if contrastive_type == 'am':
        for i, ((data, y_true), (encoded_data, _)) in enumerate(zip_loader):
            print(f'Batch {i} / {len(zip_loader)}')
            n = data.size(0)
            tiled_data = data.repeat_interleave(num_labels - 1, dim=0)
            tiled_encoded = encoded_data.repeat_interleave(num_labels - 1, dim=0)

            y_true = classifier(data.cuda()).argmax(-1)
            tiled_target = torch.tensor([[i for i in range(num_labels) if i != y_true[j]]
                                         for j in range(n)])
            tiled_target = tiled_target.reshape(-1)

            x_hat = am(generator,
                       classifier,
                       tiled_data,
                       tiled_target,
                       z_init=tiled_encoded,
                       disable_tqdm=False,
                       **kwargs)
            examples.append(x_hat.cpu().detach().numpy())
            example_labels.append(tiled_target.numpy())

    elif contrastive_type == 'cdeepex':
        data, encoded_data = [], []
        for (d, _), (enc_d, _) in tqdm(zip_loader, desc='Collecting data'):
            data.append(d)
            encoded_data.append(enc_d)
        data = torch.cat(data)
        encoded_data = torch.cat(encoded_data)

        print('Tiling data')
        n = data.size(0)
        tiled_data = data.repeat_interleave(num_labels - 1, dim=0)
        tiled_encoded = encoded_data.repeat_interleave(num_labels - 1, dim=0)

        y_true = classifier(data.cuda()).argmax(-1)
        tiled_target = torch.tensor([[i for i in range(num_labels) if i != y_true[j]]
                                     for j in range(n)])
        tiled_target = tiled_target.reshape(-1)

        print('Running cdeepex')
        x_hat = cdeepex(generator,
                        classifier,
                        tiled_data,
                        tiled_target,
                        num_labels,
                        z_init=tiled_encoded,
                        **kwargs)

        examples = [x_hat.cpu().detach().numpy()]
        example_labels = [tiled_target.numpy()]

    elif contrastive_type == 'xgems':
        for (data, label), (encoded_data, _) in tqdm(zip_loader):
            for target_label in range(num_labels):
                if target_label == label:
                    continue
                target_label = torch.tensor(target_label).unsqueeze(0)
                x_hat = xgems(generator,
                              classifier,
                              data,
                              target_label,
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


def get_contrastive_dataset_path(contrastive_type, dataset, classifier_net_type,
                                 generator_net_type):
    return Path(
        f'data/{contrastive_type}_{dataset}_{classifier_net_type}_{generator_net_type}.npz')


if __name__ == '__main__':
    fire.Fire()
