import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        while True:
            #Look untill the same class image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        #img0 = img0.convert("L")
        #img1 = img1.convert("L")

        if self.transform is not None:
            img0, img1 = self.transform(img0, img1)
            #img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
