import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from validity.datasets import load_datasets
from validity.util import EarlyStopping


def train_ds(net, save_path, dataset, *args, val_prop=0.1, **kwargs):
    train_ds, test_ds = load_datasets(dataset)
    train_size = int(len(train_ds) * (1 - val_prop))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [train_size, len(train_ds) - train_size])

    return train(net, save_path, train_ds, val_ds, test_ds, *args, **kwargs)


def train(net,
          save_path,
          train_set,
          val_set,
          test_set,
          opt_name='adam',
          batch_size=64,
          max_epochs=1000,
          tensorboard_path=None,
          lr=1e-3,
          l2_penalty=0.,
          max_grad_norm=None,
          lipshitz_gp=None,
          adv_steps=None,
          adv_eps=None):
    '''
    Trains a classifier

    Good values for:
    - Adversarial training (for free):
        - `optim='sgd'`
        - `adv_steps=7`
        - `adv_eps=1e-1`

    '''
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    if opt_name == 'sgd':
        opt = optim.SGD(net.parameters(), lr=lr, weight_decay=l2_penalty)
    elif opt_name == 'adam':
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_penalty)
    else:
        raise Exception(f'No valid optimizer specified: {opt_name}')

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    early_stopping = EarlyStopping({
        'args': net.get_args(),
        'state_dict': net.state_dict()
    }, save_path)
    tb_writer = None
    if tensorboard_path:
        tb_writer = SummaryWriter(tensorboard_path)

    step = 0
    _max_epochs = max_epochs if adv_steps is None else max_epochs // adv_steps
    for epoch in range(_max_epochs):
        print(f'Epoch {epoch * (adv_steps or 1)}')
        for data, labels in tqdm(train_loader):
            data = data.cuda().requires_grad_(True)
            delta = torch.zeros_like(data).cuda()
            for adv_step in range(adv_steps or 1):
                opt.zero_grad()
                if data.grad is not None:
                    data.grad.zero_()
                outputs = net(data + delta)
                loss = criterion(outputs, labels.cuda())

                if lipshitz_gp:
                    gradients = autograd.grad(outputs=loss, inputs=data, create_graph=True)[0]
                    grad_norm = gradients.square().sum((2, 3)).sqrt()
                    grad_penalty = (grad_norm - 1.).square() * lipshitz_gp

                    if tb_writer and adv_step == 0:
                        tb_writer.add_scalar('train/grad_norm', grad_norm.mean(), step)

                    loss += grad_penalty.mean()

                loss.backward()

                if tb_writer and adv_step == 0:
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels.cuda()).sum().item()
                    tb_writer.add_scalar('train/loss', loss.mean(), step)
                    tb_writer.add_scalar('train/inaccuracy', 1. - correct / total, step)

                if max_grad_norm:
                    total_norm = clip_grad_norm_(net.parameters(), max_grad_norm)

                    if tb_writer and adv_step == 0:
                        tb_writer.add_scalar('train/total_norm', total_norm)

                opt.step()

                if adv_steps:
                    with torch.no_grad():
                        delta += adv_eps * data.grad.sign_()
                        delta.clamp_(-adv_eps, adv_eps)

            step += 1
            if tb_writer:
                tb_writer.add_scalar('train/loss', loss, step)

        losses = []
        val_correct = 0.
        val_total = 0.
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                outputs = net(data.cuda())
                loss = criterion(outputs, labels.cuda())
                losses.append(loss)

                if tb_writer:
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels.cuda()).sum().item()

        loss = torch.mean(torch.tensor(loss))
        if tb_writer:
            tb_writer.add_scalar('val/loss', loss, step)
            tb_writer.add_scalar('val/inaccuracy', 1. - val_correct / val_total, step)

        if early_stopping(loss):
            break

    losses = []
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = net(data.cuda())
            loss = criterion(outputs, labels.cuda())
            losses.append(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()

    if tb_writer:
        loss = torch.mean(torch.tensor(losses))
        tb_writer.add_scalar('test/accuracy', loss)
        tb_writer.add_scalar('test/accuracy', correct / total)

    print(f'Accuracy {correct / total}')
