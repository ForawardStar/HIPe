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
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('result/', exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)

# Initialize generator and discriminator
G_network = GeneratorResNet(res_blocks=opt.n_residual_blocks)
#G_network.apply(weights_init)

if cuda:
    G_network = G_network.cuda()

# Load pretrained models
G_network.load_state_dict(torch.load('saved_models/smoothing/G_AB_400.pth'))

# Initialize weights
#G_network.apply(weights_init_normal)


# Optimizers

# Learning rate update schedulers

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
              transforms.ToTensor()]

# Training data loader
dataloader = DataLoader(ImageDataset("/home/root_path", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
# Test data loader
#val_dataloader = DataLoader(ImageDataset("/home/root_path", transforms_=transforms_, unaligned=True, mode='test'),
#                        batch_size=8, shuffle=True, num_workers=1)

def sample_images(batches_done, filename, _real_A, _fake_B, _mask):

    """Saves a generated sample from the test set"""

    real_A = _real_A.unsqueeze(0)

    fake_B = _fake_B.unsqueeze(0) 
    mask = _mask.unsqueeze(0) 



    #edge_detection1 = F.conv2d(fake_B, weight1, stride=1, padding=1).detach()
    #edge_detection2 = F.conv2d(fake_B, weight2, stride=1, padding=1).detach()
    #edge_detection = (torch.abs(edge_detection1) + torch.abs(edge_detection2)) / 2 
    #edge_detection_threshold = torch.where(edge_detection > 2.5,torch.ones(edge_detection.shape).cuda(),torch.zeros(edge_detection.shape).cuda())

    img_sample1 = torch.cat((real_A.data, fake_B.data[:,0:3,:,:], mask.data[:,0:3,:,:]), 0)

    #save_image(img_sample1, 'testresult_gt_siggraphasis/%s_1.png' % batches_done, nrow=5, normalize=False)

    img_sample2 = torch.cat((real_A.data, fake_B.data[:,3:6,:,:], mask.data[:,3:6,:,:]), 0)
    #save_image(img_sample2, 'result/%s_2.png' % batches_done, nrow=5, normalize=True)
    img_sample3 = torch.cat((real_A.data, fake_B.data[:,6:9,:,:], mask.data[:,6:9,:,:]), 0)
    #save_image(img_sample3, 'result/%s_3.png' % batches_done, nrow=5, normalize=True)
    img_sample4 = torch.cat((real_A.data, fake_B.data[:,9:12,:,:], mask.data[:,9:12,:,:]), 0)
    #save_image(img_sample4, 'result/%s_4.png' % batches_done, nrow=5, normalize=True)
    #save_image(torch.cat([img_sample1,img_sample2,img_sample3,img_sample4], dim=2), 'testresult/%s.png' % filename.split('.')[0], nrow=5, normalize=False)

    #save_image(fake_B.data[:,0:3,:,:], "/home/vector/fuyuanbin/smoothing_rnn/smooth_Images2/%s_0.png" % filename.split(".")[0], nrow=1, normalize=False)


    save_image(fake_B.data[:,0:3,:,:], "/home/vector/fuyuanbin/smoothing_rnn/smooth_Images/%s_0.png" % filename.split(".")[0], nrow=1, normalize=False)
    save_image(fake_B.data[:,3:6,:,:], "/home/vector/fuyuanbin/smoothing_rnn/smooth_Images/%s_1.png" % filename.split(".")[0], nrow=1, normalize=False)

    save_image(fake_B.data[:,6:9,:,:], "/home/vector/fuyuanbin/smoothing_rnn/smooth_Images/%s_2.png" % filename.split(".")[0], nrow=1, normalize=False)
    save_image(fake_B.data[:,9:12,:,:], "/home/vector/fuyuanbin/smoothing_rnn/smooth_Images/%s_3.png" % filename.split(".")[0], nrow=1, normalize=False) 
    #print("edge_path: ", edge_path)

# ----------
#  Training
# ----------

prev_time = time.time()
#G_network.eval()
for epoch in range(opt.epoch, 2):
    for i, batch in enumerate(dataloader):

        _input_edge = Variable(batch['edge'].type(Tensor))
        _input_image = Variable(batch['img'].type(Tensor))
        _filename = batch['filename']

        print("_input_image shape:  {}  ,  _input_edge: {}".format( _input_image.shape, _input_edge.shape))
        generated_images = G_network(_input_image ,_input_edge, 1)

        # ------------------
        #  Train Generators
        # ------------------
        #print("filename:  ", _filename)       

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        save_image(generated_images.data[:,0:3,:,:], "testresult/%s_0.png" % _filename[0].split(".")[0], nrow=1, normalize=False)
        save_image(generated_images.data[:,3:6,:,:], "testresult/%s_1.png" % _filename[0].split(".")[0], nrow=1, normalize=False)
        save_image(generated_images.data[:,6:9,:,:], "testresult/%s_2.png" % _filename[0].split(".")[0], nrow=1, normalize=False)
        save_image(generated_images.data[:,9:12,:,:], "testresult/%s_3.png" % _filename[0].split(".")[0], nrow=1, normalize=False)

        del _filename, _input_image, generated_images, _input_edge

    # Update learning rates


