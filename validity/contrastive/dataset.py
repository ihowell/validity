from pathlib import Path
import time

import fire
import torch
import numpy as np

from tqdm import tqdm

from validity.classifiers.load import load_cls
from validity.datasets import load_datasets
from validity.generators.load import load_gen, load_encoded_ds, get_encoded_ds_path
from validity.util import get_executor, TiledDataset

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
                             classifier_id=None,
                             subset=None,
                             **kwargs):
    assert contrastive_type in ['am', 'xgems', 'cdeepex']

    executor = get_executor()
    jobs = []
    with executor.batch():
        for shard_idx in range(shards):
            shard_path = _get_contrastive_dataset_shard_path(contrastive_type,
                                                             dataset,
                                                             classifier_net_type,
                                                             generator_net_type,
                                                             subset,
                                                             shard_idx,
                                                             shards,
                                                             classifier_id=classifier_id)
            if not Path(shard_path).exists():
                jobs.append(
                    executor.submit(_make_contrastive_dataset_job,
                                    contrastive_type,
                                    dataset,
                                    classifier_net_type,
                                    classifier_weights_path,
                                    generator_net_type,
                                    generator_weights_path,
                                    shard_idx,
                                    shards,
                                    batch_size=batch_size,
                                    data_root=data_root,
                                    cuda_idx=cuda_idx,
                                    seed=seed,
                                    subset=subset,
                                    classifier_id=classifier_id,
                                    **kwargs))
    [job.results() for job in jobs]

    examples = []
    example_labels = []
    for i in range(shards):
        shard_path = _get_contrastive_dataset_shard_path(contrastive_type,
                                                         dataset,
                                                         classifier_net_type,
                                                         generator_net_type,
                                                         subset,
                                                         i,
                                                         shards,
                                                         classifier_id=classifier_id)
        _file = np.load(shard_path)
        examples.append(_file['arr_0'])
        example_labels.append(_file['arr_1'])
    examples = np.concatenate(examples)
    example_labels = np.concatenate(example_labels)

    save_path = get_contrastive_dataset_path(contrastive_type,
                                             dataset,
                                             classifier_net_type,
                                             generator_net_type,
                                             subset=subset,
                                             classifier_id=classifier_id)
    Path(save_path).parent.mkdir(exist_ok=True)
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
                                  classifier_id=None,
                                  subset=None,
                                  **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_labels = 10

    print('Loading datasets')
    _, test_ds = load_datasets(dataset)
    encoded_test_ds = None
    if get_encoded_ds_path(dataset, generator_net_type).exists():
        print('Using cached encoded dataset')
        encoded_test_ds = load_encoded_ds(dataset, generator_net_type)
    else:
        print('No cached encoded dataset found')

    if subset:
        test_ds = torch.utils.data.Subset(test_ds, range(subset))
        if encoded_test_ds:
            encoded_test_ds = torch.utils.data.Subset(encoded_test_ds, range(subset))

    print('Sharding dataset')
    n = len(test_ds)
    shard_lower = (n * shard_idx) // shards
    shard_upper = (n * (shard_idx + 1)) // shards
    test_ds = torch.utils.data.Subset(test_ds, range(shard_lower, shard_upper))
    tiled_ds = TiledDataset(test_ds, num_labels)
    tiled_loader = torch.utils.data.DataLoader(tiled_ds, batch_size=batch_size, shuffle=False)

    if encoded_test_ds:
        encoded_test_ds = torch.utils.data.Subset(encoded_test_ds,
                                                  range(shard_lower, shard_upper))
        encoded_tiled_ds = TiledDataset(encoded_test_ds, num_labels)
        encoded_test_loader = torch.utils.data.DataLoader(encoded_tiled_ds,
                                                          batch_size=batch_size,
                                                          shuffle=False)
        encoded_iter = iter(encoded_test_loader)

    print('Loading classifier')
    classifier = load_cls(classifier_net_type, classifier_weights_path, dataset)
    classifier = classifier.cuda()
    classifier.eval()

    print('Loading generator')
    generator = load_gen(generator_net_type, generator_weights_path, dataset)
    generator = generator.cuda()
    generator.eval()

    examples = []
    example_labels = []

    start = time.time()

    if contrastive_type == 'am':
        for i, (data, y_probe) in enumerate(tiled_loader):
            print(f'Batch {i} / {len(tiled_loader)}')
            encoded_data = None
            if encoded_test_ds:
                encoded_data = next(encoded_iter)[0]

            x_hat = am(generator,
                       classifier,
                       data,
                       y_probe,
                       z_init=encoded_data,
                       disable_tqdm=False,
                       **kwargs)
            examples.append(x_hat.cpu().detach().numpy())
            example_labels.append(y_probe.numpy())

    elif contrastive_type == 'cdeepex':
        data = []
        y_probe = []
        encoded_data = [] if encoded_test_ds else None
        for d, y in tqdm(tiled_loader, desc='Collecting data'):
            data.append(d)
            y_probe.append(y)
            if encoded_test_ds:
                encoded_data.append(next(encoded_iter)[0])
        data = torch.cat(data)
        y_probe = torch.cat(y_probe)
        if encoded_test_ds:
            encoded_data = torch.cat(encoded_data)

        print('Running cdeepex')
        x_hat = cdeepex(generator,
                        classifier,
                        data,
                        y_probe,
                        num_labels,
                        z_init=encoded_data,
                        batch_size=batch_size,
                        **kwargs)

        examples = [x_hat.cpu().detach().numpy()]
        example_labels = [y_probe.numpy()]

    elif contrastive_type == 'xgems':
        for i, (data, y_probe) in enumerate(tiled_loader):
            print(f'Batch {i} / {len(tiled_loader)}')
            encoded_data = None
            if encoded_test_ds:
                encoded_data = next(encoded_iter)[0]

            x_hat = xgems(generator,
                          classifier,
                          data,
                          y_probe,
                          z_init=encoded_data,
                          disable_tqdm=False,
                          **kwargs)
            examples.append(x_hat.cpu().detach().numpy())
            example_labels.append(y_probe.numpy())

    finish = time.time()
    print(f'Time to complete: {finish - start:.1f} sec')

    examples = np.concatenate(examples)
    example_labels = np.concatenate(example_labels)

    Path('data/tmp').mkdir(exist_ok=True, parents=True)
    shard_path = _get_contrastive_dataset_shard_path(contrastive_type,
                                                     dataset,
                                                     classifier_net_type,
                                                     generator_net_type,
                                                     subset,
                                                     shard_idx,
                                                     shards,
                                                     classifier_id=classifier_id)
    np.savez(shard_path, examples, example_labels)


def _get_contrastive_dataset_shard_path(contrastive_type,
                                        dataset,
                                        classifier_net_type,
                                        generator_net_type,
                                        subset,
                                        shard_idx,
                                        shards,
                                        classifier_id=None):
    if subset:
        subset = int(subset)
    save_path = f'data/tmp/{contrastive_type}_{dataset}_{classifier_net_type}'
    if classifier_id:
        save_path += f'_{classifier_id}'
    save_path += f'_{generator_net_type}_sub_{subset}_{shard_idx}_{shards}.npz'
    return save_path


def get_contrastive_dataset_path(contrastive_type,
                                 dataset,
                                 classifier_net_type,
                                 generator_net_type,
                                 classifier_id=None,
                                 subset=None):
    if subset:
        subset = int(subset)
    file_name = f'{contrastive_type}_{dataset}_{classifier_net_type}'
    if classifier_id:
        file_name += f'_{classifier_id}'
    file_name += f'_{generator_net_type}_sub_{subset}'
    return Path(f'data/{file_name}.npz')


if __name__ == '__main__':
    fire.Fire()
