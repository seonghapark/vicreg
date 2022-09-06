# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
import torch

import torch.nn.functional as F
import torch.distributed as dist

import resnet

from dataset import BasicDataset

import numpy as np

import gc
import glob

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Data
    parser.add_argument("--data-dir", type=str, help="path to dataset")

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser.parse_args()


class Main():
    def __init__(self, args):
        args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node
        self.gpu = 0
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=0
        )

        torch.cuda.set_device(self.gpu)
        torch.backends.cudnn.benchmark = True

        self.model = VICReg(args).cuda(self.gpu)
        self.model.cuda(self.gpu)
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    def run(self, args, train_loader, val_loader):
        start_time = time.time()
        # evaluate
        self.model.eval()
        for step, (x, y) in enumerate(zip(train_loader, val_loader), len(val_loader)):
            x = x.cuda(self.gpu, non_blocking=True)
            y = y.cuda(self.gpu, non_blocking=True)

            loss, repr_loss, std_loss, cov_loss = self.model.forward(x, y)
            #print('repr_loss', (args.sim_coeff * repr_loss * 100).item(),
            #      'std_loss', (args.std_coeff * std_loss).item(),
            #      'cov_loss', (args.cov_coeff * cov_loss).item())

        return (args.sim_coeff * repr_loss * 100).item()


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

        self.state_dict = torch.load(args.pretrained, map_location="cpu")
        if "model" in self.state_dict:
            self.state_dict = self.state_dict["model"]
            self.state_dict = {
                key.replace("module.backbone.", ""): value
                for (key, value) in self.state_dict.items()
            }
        self.backbone.load_state_dict(self.state_dict, strict=False)
        self.backbone.requires_grad_(False)

        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass







def same_folder(args, files):
    for i in range(len(files)):
        if i != len(files)-1:
            args.data_dir1 = files[i]
            args.data_dir2 = files[i+1]
        else:
            args.data_dir1 = files[i]
            args.data_dir2 = files[i-1]

        train_dataset = BasicDataset(args.data_dir1)
        val_dataset = BasicDataset(args.data_dir2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)


        values = []
        val = np.asarray(main.run(args, train_loader, val_loader))
        values.append(val)

    return values


def diff_folders(args, f1, f2):
    if len(f1) <= len(f2):
        files1 = f1
        files2 = f2
    else:
        files1 = f2
        files2 = f1

    for i in range(len(files1)):
        args.data_dir1 = files1[i]
        args.data_dir2 = files2[i]

        train_dataset = BasicDataset(args.data_dir1)
        val_dataset = BasicDataset(args.data_dir2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)


        values = []
        val = np.asarray(main.run(args, train_loader, val_loader))
        values.append(val)

    return values


def with_the_folders(args):
    name = args.data_dir1 + '*'
    files1 = sorted(glob.glob(name))
    name = args.data_dir2 + '*'
    files2 = sorted(glob.glob(name))

    if args.data_dir1 == args.data_dir2:
        values = same_folder(args, files1)
    else:
        values = diff_folders(args, files1, files2)

    return values


if __name__ == "__main__":
    args = get_arguments()
    main = Main(args)

    folders = sorted(os.listdir(args.data_dir))

    limit = 50

    with open('meanrepr.txt', 'w') as f:
        for i in folders:
            for j in folders:
                if int(i) < limit and int(j) < limit:
                    args.data_dir1 = args.data_dir + i + '/'
                    args.data_dir2 = args.data_dir + j + '/'

                    values = with_the_folders(args)
                    v = np.asarray(values)
                    l = str(i) + ',' + str(j) + ',' + str(v.mean()) + '\n'
                    f.write(l)
                    print(i, j, v.mean())
