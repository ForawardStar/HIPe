import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from test_datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from loss_function import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="BSDS500", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion = MyLoss().cuda()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)

# Initialize generator and discriminator

G_network = GeneratorResNet()
G_network.apply(weights_init)

if cuda:
    G_network = G_network.cuda()

# Load pretrained models
G_network.load_state_dict(torch.load('saved_models/BSDS500/G_AB_100.pth'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [transforms.ToTensor()]

#Data loader
dataloader = DataLoader(ImageDataset("/home/root_path", transforms_=transforms_, unaligned=True),
                        batch_size=1, shuffle=True, num_workers=1)

# ----------
#  Training
# ---------
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        filename = batch['filename']
        input_image = Variable(batch['img'].type(Tensor))
        
        mask_images = G_network(input_image)
      
        print("filename:   ", filename)
        edge_path_formal = "testresult/"

        save_image(mask_images.data[:, 0:3, :, :], edge_path_formal + filename[0].split(".")[0] + "_0.png", nrow=1,
                   normalize=False)
        save_image(mask_images.data[:, 3:6, :, :], edge_path_formal + filename[0].split(".")[0] + "_1.png", nrow=1,
                   normalize=False)
        save_image(mask_images.data[:, 6:9, :, :], edge_path_formal + filename[0].split(".")[0] + "_2.png", nrow=1,
                   normalize=False)
        save_image(mask_images.data[:, 9:12, :, :], edge_path_formal + filename[0].split(".")[0] + "_3.png", nrow=1,
                   normalize=False)
        
        # --------------
        #  Log Progress
        # --------------
        del mask_images, input_image, filename
