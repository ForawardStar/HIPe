import torch.nn as nn
import torch.nn.functional as F
import torch
#import pytorch_colors as colors
import copy

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterionMSE = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionCLS = nn.BCELoss()
        self.a = 1.5 

        self.u = 20 

        self.b = 10.5 
        self.P1 = 0.5
        self.P2 = 0.2
        self.P3 = 0.05
        self.P4 = 0.01
        self.e = 0.005

        self.sobel_kernelh = torch.cuda.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).expand(4,1,3,3)
        self.sobel_kernelw = torch.cuda.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(4,1,3,3)
        self.weighth = nn.Parameter(data=self.sobel_kernelh, requires_grad=False)
        self.weightw = nn.Parameter(data=self.sobel_kernelw, requires_grad=False)

        self.dim = (1,1,1,1)

    def RGB2GRAY(self, tensor):
        # TODO: make efficient
        R1 = tensor[:,0:1,:,:]
        G1 = tensor[:,1:2,:,:]
        B1 = tensor[:,2:3,:,:]
        tensor1=0.299*R1+0.587*G1+0.114*B1

        R2 = tensor[:,3:4,:,:]
        G2 = tensor[:,4:5,:,:]
        B2 = tensor[:,5:6,:,:]
        tensor2=0.299*R2+0.587*G2+0.114*B2

        R3 = tensor[:,6:7,:,:]
        G3 = tensor[:,7:8,:,:]
        B3 = tensor[:,8:9,:,:]
        tensor3=0.299*R3+0.587*G3+0.114*B3

        R4 = tensor[:,9:10,:,:]
        G4 = tensor[:,10:11,:,:]
        B4 = tensor[:,11:12,:,:]
        tensor4=0.299*R4+0.587*G4+0.114*B4

        tensor = torch.cat([tensor1,tensor2,tensor3,tensor4],dim=1)
        return tensor

    def NMS(self, dx, dy):
        M = dx + dy
        d = copy.copy(M)
        H, W = M.shape
        NMS = copy.copy(d)
        NMS[0, :] = NMS[H-1, :] = NMS[:, 0] = NMS[:, W-1] = 0

        for i in range(1, H-1):
            for j in range(1, W-1):

                if M[i, j] == 0:
                    NMS[i, j] = 0

                else: 
                    gradX = dx[i, j] # the gradient of the point in the x-direction 
                    gradY = dy[i, j] # the gradient of the point in the y-direction
                    gradTemp = d[i, j] # the gradient of the point
                # if the gradient in the y-direction is larger, meaning the direction of dervatives tend to y component
                    if torch.abs(gradY) > torch.abs(gradX):
                        weight = torch.abs(gradX) / torch.abs(gradY) #weight 
                        grad2 = d[i-1, j]
                        grad4 = d[i+1, j]
                    # if the signs of x and y direction are consistent
                    # g1 g2
                    #    c
                    #    g4 g3
                        if gradX * gradY > 0:
                            grad1 = d[i-1, j-1]
                            grad3 = d[i+1, j+1]

                    # if the signs of x and y direction are opposite 

                    #    g2 g1
                    #    c
                    # g3 g4
                        else:
                            grad1 = d[i-1, j+1]
                            grad3 = d[i+1, j-1]

                # if the gradient in the x-direction is larger 
                    else:
                        weight = torch.abs(gradY) / torch.abs(gradX)
                        grad2 = d[i, j-1]
                        grad4 = d[i, j+1]

                    #      g3
                    # g2 c g4
                    # g1
                        if gradX * gradY > 0:

                            grad1 = d[i+1, j-1]
                            grad3 = d[i-1, j+1]

                    # g1
                    # g2 c g4
                    #      g3
                        else:
                            grad1 = d[i-1, j-1]
                            grad3 = d[i+1, j+1]

                # interpolate the gradient using grad1-grad4
                    gradTemp1 = weight * grad1 + (1 - weight) * grad2
                    gradTemp2 = weight * grad3 + (1 - weight) * grad4

                #  the point is the local maxium and may be the edge point
                    if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                       NMS[i, j] = gradTemp

                    else:
                    # not the edge point
                        NMS[i, j] = 0

        return NMS

    def forward(self, targetimg, masks, gt_edge, epoch):
        #u = self.u * math.log(epoch + 2)
        # a = min(self.a * math.log(epoch + 2),0.04)
        #genimgs_L = RGB2GRAY(genimgs)  
        targetimgs = torch.cat([targetimg, targetimg, targetimg, targetimg], dim=1)
        #gt_edges = torch.cat([gt_edge, gt_edge, gt_edge, gt_edge], dim=1)
        batch_size = targetimgs.size()[0]

        channels = targetimgs.size()[1]
        h_img = targetimgs.size()[2]
        w_img = targetimgs.size()[3]

        gradient_targetimg_h = F.pad(torch.abs(targetimg[:, :, 1:, :] - targetimg[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_targetimg_w = F.pad(torch.abs(targetimg[:, :, :, 1:] - targetimg[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        gradient_targetimg = (gradient_targetimg_h + gradienth_targetimg_w)

        # loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4
        masks_gradient_1 = torch.sum(masks[:, 0:3, :, :]) / (h_img * w_img * channels * batch_size)
        masks_gradient_2 = torch.sum(masks[:, 3:6, :, :]) / (h_img * w_img * channels * batch_size)
        masks_gradient_3 = torch.sum(masks[:, 6:9, :, :]) / (h_img * w_img * channels * batch_size)
        masks_gradient_4 = torch.sum(masks[:, 9:12, :, :]) / (h_img * w_img * channels * batch_size)

        loss2_1 = torch.abs(masks_gradient_1 - self.P1)**2
        loss2_2 = torch.abs(masks_gradient_2 - self.P2)**2
        loss2_3 = torch.abs(masks_gradient_3 - self.P3)**2
        loss2_4 = torch.abs(masks_gradient_4 - self.P4)**2
        loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4

        #compute 3th loss
        #gradient_genimg = (gradient_genimg_h + gradienth_genimg_w)
        gradient_targetimg1 = gradient_targetimg + gt_edge - 0.4 * torch.abs(1 - gt_edge) * gradient_targetimg
        gradient_targetimg2 = gradient_targetimg1 - 0.5 * torch.abs(1 - gt_edge) * gradient_targetimg1
        gradient_targetimg3 = gradient_targetimg2 - 0.8 * torch.abs(1 - gt_edge) * gradient_targetimg2
        masked_relative1 = masks[:, 0:3, :, :] * torch.abs(gradient_targetimg1 - 1) + 20 * torch.abs(1 - masks[:, 0:3, :, :]) * gradient_targetimg1
        masked_relative2 = masks[:, 3:6, :, :] * torch.abs(gradient_targetimg2 - 1) + 20 * torch.abs(1 - masks[:, 3:6, :, :]) * gradient_targetimg2
        masked_relative3 = masks[:, 6:9, :, :] * torch.abs(gradient_targetimg3 - 1) + 15 * torch.abs(1 - masks[:, 6:9, :, :]) * gradient_targetimg3
        masked_relative4 = masks[:, 9:12, :, :] * torch.abs(gt_edge - 1) + 15 * torch.abs(1 - masks[:, 9:12, :, :]) * gt_edge
        masked_relative = masked_relative1 + masked_relative2 + masked_relative3 + masked_relative4
        #masked_relative_w = masks * torch.abs(gradienth_genimg_w - 1) + 40 * torch.abs(1 - masks) * gradienth_genimg_w
        loss4 = torch.mean(masked_relative)

        #compute total loss
        #totalloss = loss1 + self.u*loss2 + b*loss3 + a*loss4
        totalloss = self.a*loss4
        print("self.u: {}  ,  b: {}  ,  a: {}".format(self.u,self.b,self.a))
        print("epoch:{}  ,  loss2:{}  ,  loss4:{}  ,  totalloss:{}".format(epoch,loss2,loss4,totalloss))

        return totalloss