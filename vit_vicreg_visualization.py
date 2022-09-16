from pathlib import Path
import glob
import time
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import vit_utils
import vision_transformer as vits

from dataset import BasicDataset

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone =  vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        self.embedding = 384

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



class Main():
    def __init__(self, args):
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = 'cpu'
        self.model = VICReg(args).to(device)
        self.model.to(device)
        self.model.eval()

    def run(self, args, train_loader, val_loader):
        start_time = time.time()
        # evaluate
        self.model.eval()
        for step, (x, y) in enumerate(zip(train_loader, val_loader), len(val_loader)):
            #x = x.cuda(self.gpu, non_blocking=True)
            #y = y.cuda(self.gpu, non_blocking=True)

            loss, repr_loss, std_loss, cov_loss = self.model.forward(x, y)
            print('repr_loss', (args.sim_coeff * repr_loss * 100).item(),
                  'std_loss', (args.std_coeff * std_loss).item(),
                  'cov_loss', (args.cov_coeff * cov_loss).item())

        return (args.sim_coeff * repr_loss * 100).item()


    def return_model(self):
        return self.model

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
            train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')


    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument("--mlp", default="1536-1536-1536",
    #parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')


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
    parser.add_argument("--exp-dir", default="./exp/", type=Path, metavar="DIR",
                         help="path to checkpoint directory")

    args = parser.parse_args()

    main = Main(args)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)


    folders = sorted(os.listdir(args.data_dir))

    lower_limit = 50
    upper_limit = 100

    outputfile = open('meanrepr_vit.txt', 'a', buffering=1)
    for i in folders:
        for j in folders:
            if int(i) < upper_limit and int(i) > lower_limit and int(j) > lower_limit and int(j) < upper_limit:
                args.data_dir1 = args.data_dir + i + '/'
                args.data_dir2 = args.data_dir + j + '/'

                values = with_the_folders(args)
                v = np.asarray(values)
                l = str(i) + ',' + str(j) + ',' + str(v.mean())

                print(json.dumps(l))
                print(json.dumps(l), file=outputfile)
