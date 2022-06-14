import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.misc


class SALICON(data.Dataset):

    def __init__(self, stimuli_train_path, stimuli_val_path, gt_train_path, gt_val_path, augment=None, transform_h=None, transform_l=None, target=None, mode='train'):
        self.gt_train_path = gt_train_path
        self.gt_val_path = gt_val_path
        self.mode = mode
        self.augment = augment
        self.transform_h = transform_h
        self.transform_l = transform_l
        self.target = target

        if (mode == 'train'):
            self.data_list = sorted(glob.glob(stimuli_train_path + '*'))
            self.gt_list = sorted(glob.glob(gt_train_path + '*'))
            print('Total training samples are {}'.format(len(self.data_list)))
        else:
            self.data_list = sorted(glob.glob(stimuli_val_path + '*'))
            self.gt_list = sorted(glob.glob(gt_val_path + '*'))
            print('Total validating samples are {}'.format(len(self.data_list)))


    def __getitem__(self, idx):
        data_stimuli_path = self.data_list[idx]
        data_gt_path = self.gt_list[idx]

        data_stimuli = Image.open(data_stimuli_path)
        data_gt = Image.open(data_gt_path)

        if (self.augment != None):
            data_stimuli, data_gt = self.augment(data_stimuli, data_gt)

        if (self.transform_h != None):
            data_stimuli_h = self.transform_h(data_stimuli)

        if (self.transform_l != None):
            data_stimuli_l = self.transform_l(data_stimuli)

        if (self.target != None):
            data_gt = self.target(data_gt)

        return data_stimuli_h, data_stimuli_l, data_gt


    def __len__(self):
        return len(self.data_list)
