"""Standard CIFAR-10 ResNet-20 (He et al., 2016, option-A shortcuts) and a CPU training run.

Trains from the local CIFAR-10 copy in cnn/data; the checkpoint is used for the
bit-exact approximate-accumulator inference study (cnn_accumulate.py).
"""
import argparse
import os
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ACC_DATASET=cifar100 selects the pre-registered CIFAR-100 replication; CIFAR-10 is the default
DATASET = os.environ.get("ACC_DATASET", "cifar10")
if DATASET == "cifar100":
    DATA = Path(os.environ.get("CIFAR100_DIR", r"D:/accrate_data/cifar-100-python"))
    CLASSES, _DEFAULT_CKPT = 100, "resnet20_cifar100.pt"
    MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
    STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)
else:
    DATA = Path(__file__).resolve().parents[2] / "cnn" / "data" / "cifar-10-batches-py"
    CLASSES, _DEFAULT_CKPT = 10, "resnet20_cifar10.pt"
    MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CKPT = Path(__file__).with_name(os.environ.get("RESNET_CKPT", _DEFAULT_CKPT))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.pad = cout - cin if (stride != 1 or cin != cout) else 0
        self.stride = stride

    def shortcut(self, x):
        if not self.pad and self.stride == 1:
            return x
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, n=3, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            for i in range(n):
                layers.append(Block(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load_cifar(train):
    if DATASET == "cifar100":
        files, key = ["train" if train else "test"], b"fine_labels"
    else:
        files, key = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"], b"labels"
    xs, ys = [], []
    for f in files:
        d = pickle.load(open(DATA / f, "rb"), encoding="bytes")
        xs.append(d[b"data"]); ys += d[key]
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(x), torch.tensor(ys)


def augment(x, g):
    pad = F.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2,), generator=g).tolist()
    x = pad[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(x.shape[0], generator=g) < 0.5
    x[flip] = x[flip].flip(3)
    return x


def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def train(epochs, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = load_cifar(True)
    xte, yte = load_cifar(False)
    model = ResNet20(classes=CLASSES)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    bs = 128
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.2, total_steps=steps, pct_start=0.25)
    log = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        t0 = time.time()
        for b in range(len(xtr) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            # crop offsets drawn per batch, flips per image
            xb = augment(xtr[idx], g)
            loss = F.cross_entropy(model(xb), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = evaluate(model, xte, yte)
        log.append({"epoch": ep + 1, "test_acc": acc, "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        torch.save({"state_dict": model.state_dict(), "log": log, "seed": seed}, CKPT)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    torch.set_num_threads(20)
    args = ap.parse_args()
    train(args.epochs, args.seed)
