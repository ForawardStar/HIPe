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
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from loss_function import MyLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=701, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="BSDS500", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=1, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('result/', exist_ok=True)
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

optimizer_G = torch.optim.RMSprop(filter(lambda p: p.requires_grad, G_network.parameters()),lr=opt.lr, alpha=0.9)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [#transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
              transforms.ToTensor()]

# Training data loader
dataloader = DataLoader(ImageDataset("/home/root_path", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Val data loader
val_dataloader = DataLoader(ImageDataset("/home/root_path", transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=1, shuffle=True, num_workers=1)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    val_batch = next(iter(val_dataloader))

    _real_A = Variable(val_batch['img'].type(Tensor))
    _edge = Variable(val_batch['edge'].type(Tensor))
    _mask = G_network(_real_A)

    real_A = _real_A[0].unsqueeze(0)
    edge = _edge[0].unsqueeze(0)
    #fake_B = _fake_B[0].unsqueeze(0)
    mask = _mask[0].unsqueeze(0)

    #edge_detection1 = F.conv2d(fake_B, weight1, stride=1, padding=1).detach()
    #edge_detection2 = F.conv2d(fake_B, weight2, stride=1, padding=1).detach()
    #edge_detection = (torch.abs(edge_detection1) + torch.abs(edge_detection2)) / 2 
    #edge_detection_threshold = torch.where(edge_detection > 2.5,torch.ones(edge_detection.shape).cuda(),torch.zeros(edge_detection.shape).cuda())
    img_sample1 = torch.cat((real_A.data, edge, mask.data[:,0:3,:,:]), 0)
    #save_image(img_sample1, 'result/%s_1.png' % batches_done, nrow=5, normalize=True)
    img_sample2 = torch.cat((real_A.data, edge, mask.data[:,3:6,:,:]), 0)
    #save_image(img_sample2, 'result/%s_2.png' % batches_done, nrow=5, normalize=True)
    img_sample3 = torch.cat((real_A.data, edge, mask.data[:,6:9,:,:]), 0)
    #save_image(img_sample3, 'result/%s_3.png' % batches_done, nrow=5, normalize=True)
    img_sample4 = torch.cat((real_A.data, edge, mask.data[:,9:12,:,:]), 0)
    #save_image(img_sample4, 'result/%s_4.png' % batches_done, nrow=5, normalize=True)
    save_image(torch.cat([img_sample1,img_sample2,img_sample3,img_sample4], dim=2), 'result/%s.png' % batches_done, nrow=5, normalize=False)
# ----------
#  Training
# ----------
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        gt_edge = Variable(batch['edge'].type(Tensor))
        input_image = Variable(batch['img'].type(Tensor))
        mask_images = G_network(input_image)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # Identity loss
        loss = criterion(input_image,mask_images,gt_edge,epoch)

        loss.backward()

        optimizer_G.step()
        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_network.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))