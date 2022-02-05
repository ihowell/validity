import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from validity.util import EarlyStopping


def standard_train(net,
                   save_path,
                   train_set,
                   val_set,
                   test_set,
                   batch_size,
                   max_epochs=1000,
                   lr=1e-3):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    early_stopping = EarlyStopping(net.state_dict(), save_path)

    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')
        for data, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = net(data.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

        losses = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                outputs = net(data.cuda())
                loss = criterion(outputs, labels.cuda())
                losses.append(loss)

        loss = torch.mean(torch.tensor(loss))
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

    print(f'Accuracy {correct / total}')


def adversarial_train_for_free(net,
                               save_path,
                               train_set,
                               val_set,
                               test_set,
                               batch_size=64,
                               max_epochs=1000,
                               m=7,
                               writer=None,
                               lr=1e-3,
                               eps=1e-1):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    opt = optim.SGD(net.parameters(), lr=lr)
    early_stopping = EarlyStopping(net.state_dict(), save_path)

    step = 0
    for epoch in range(max_epochs // m):
        print(f'Epoch {epoch * m}')
        for data, labels in tqdm(train_loader):
            data = data.cuda().requires_grad_(True)
            delta = torch.zeros_like(data).cuda()
            for i in range(m):
                opt.zero_grad()
                if data.grad is not None:
                    data.grad.zero_()
                outputs = net(data + delta)
                loss = criterion(outputs, labels.cuda())
                loss.backward()

                opt.step()
                with torch.no_grad():
                    delta += eps * data.grad.sign_()
                    delta.clamp_(-eps, eps)

            step += 1
            if writer:
                writer.add_scalar('train/loss', loss, step)

        losses = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                outputs = net(data.cuda())
                loss = criterion(outputs, labels.cuda())
                losses.append(loss)

        loss = torch.mean(torch.tensor(loss))
        if writer:
            writer.add_scalar('val/loss', loss, step)
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

    if writer:
        loss = torch.mean(torch.tensor(losses))
        writer.add_scalar('test/accuracy', loss)
        writer.add_scalar('test/accuracy', correct / total)

    print(f'Accuracy {correct / total}')
