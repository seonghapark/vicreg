from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir   = imgs_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #self.ids = [imgs_dir]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize,
            ]
        )

        img_tr = transform(pil_img)
        return img_tr

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = sorted(glob(self.imgs_dir + idx + '*'))
        #img_file = self.imgs_dir
        #print(img_file)

        #assert len(img_file) == 1, \
        #    f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])
        #img = Image.open(img_file)
        img = self.preprocess(img)

        return img
