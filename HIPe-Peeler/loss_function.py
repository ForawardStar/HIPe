import torch.nn as nn
import torch.nn.functional as F
import torch
#import pytorch_colors as colors

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

#vgg = models.vgg19(pretrained=True).features
#print("vgg:  ",vgg)
#vgg = vgg.cuda()


#content_layers_default = ['conv_2']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterionMSE = nn.MSELoss()
        self.criterionL1 = nn.SmoothL1Loss()
        self.criterionCLS = nn.BCELoss()
        self.a = 0.4
        self.u = 4
        self.b = 45 

        self.P1 = 0.6
        self.P2 = 0.3
        self.P3 = 0.15
        self.P4 = 0.08
        self.e = 0.005

        self.gauss_kernel = torch.cuda.FloatTensor([[0.0947416, 0.118318, 0.0947416], [0.118318, 0.147761, 0.118318], [0.0947416, 0.118318, 0.0947416]]).unsqueeze(0).unsqueeze(0)
        #self.sobel_kernel2 = torch.cuda.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]]).expand(12,1,3,3) 
        #self.sobel_kernel3 = torch.cuda.FloatTensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]]).expand(12,1,3,3)

        #self.sobel_kernel4 = torch.cuda.FloatTensor([[0, 0, 0], [1, -1, 0], [0, 0, 0]]).expand(12,1,3,3)
        self.weight = nn.Parameter(data=self.gauss_kernel, requires_grad=False)
        #self.weight2 = nn.Parameter(data=self.sobel_kernel2, requires_grad=False)
        #self.weight3 = nn.Parameter(data=self.sobel_kernel3, requires_grad=False)
        #self.weight4 = nn.Parameter(data=self.sobel_kernel4, requires_grad=False)


    def forward(self,genimgs, targetimg, targetimgs_blur, masks, canny, epoch):

        #targetimg_blur1 = torch.abs(F.conv2d(F.conv2d(targetimg[:,0:1,:,:], self.weight, padding=1), self.weight, padding=1))
        #targetimg_blur2 = torch.abs(F.conv2d(F.conv2d(targetimg[:,1:2,:,:], self.weight, padding=1), self.weight, padding=1))
        #targetimg_blur3 = torch.abs(F.conv2d(F.conv2d(targetimg[:,2:3,:,:], self.weight, padding=1), self.weight, padding=1))
        #targetimg_blur = torch.cat([targetimg_blur1, targetimg_blur2, targetimg_blur3], dim=1) 
        targetimgs = torch.cat([targetimg, targetimg, targetimg, targetimg], dim=1)
        #targetimgs_blur = torch.cat([targetimg_blur, targetimg_blur, targetimg_blur, targetimg_blur], dim=1)
        batch_size = genimgs.size()[0]

        channels = genimgs.size()[1] 
        h_img = genimgs.size()[2]
        w_img = genimgs.size()[3]
        diffimgs = targetimgs - genimgs

        #gradient_genimg_1 = torch.abs(F.conv2d(genimgs, self.weight1, padding=1, groups=12))
        #gradient_genimg_2 = torch.abs(F.conv2d(genimgs, self.weight2, padding=1, groups=12))
        #gradient_genimg_3 = torch.abs(F.conv2d(genimgs, self.weight3, padding=1, groups=12))
        #gradient_genimg_4 = torch.abs(F.conv2d(genimgs, self.weight4, padding=1, groups=12))
        #gradient_targetimg_1 = torch.abs(F.conv2d(targetimgs, self.weight1, padding=1, groups=12))
        #gradient_targetimg_2 = torch.abs(F.conv2d(targetimgs, self.weight2, padding=1, groups=12))
        #gradient_targetimg_3 = torch.abs(F.conv2d(targetimgs, self.weight3, padding=1, groups=12))
        #gradient_targetimg_4 = torch.abs(F.conv2d(targetimgs, self.weight4, padding=1, groups=12))
        #gradient_genimg = gradient_genimg_1 + gradient_genimg_2 + gradient_genimg_3 + gradient_genimg_4
        #gradient_targetimg = gradient_targetimg_1 + gradient_targetimg_2 + gradient_targetimg_3 + gradient_targetimg_4 

        # compute struct image gradient
        gradient_genimg_h = F.pad(torch.abs(genimgs[:, :, 1:, :] - genimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_genimg_w = F.pad(torch.abs(genimgs[:, :, :, 1:] - genimgs[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        # compute origion image gradient
        gradient_targetimg_h = F.pad(torch.abs(targetimgs[:, :, 1:, :] - targetimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_targetimg_w = F.pad(torch.abs(targetimgs[:, :, :, 1:] - targetimgs[:, :, :, :w_img - 1]),(1, 0, 0, 0))
        # compute texture image gradient
        
        #gradient_diffimg_h = F.pad(torch.abs(diffimgs[:, :, 1:, :] - diffimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        #gradienth_diffimg_w = F.pad(torch.abs(diffimgs[:, :, :, 1:] - diffimgs[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        # compute mask image gradient 
        #gradient_maskimg_h = F.pad(torch.abs(masks[:, :, 1:, :] - masks[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        #gradienth_maskimg_w = F.pad(torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        gradient_genimg = gradient_genimg_h + gradienth_genimg_w
        gradient_targetimg = gradient_targetimg_h + gradienth_targetimg_w 
        #compute 1st loss
        #save_image(torch.cat([gradient_genimg_h.data[:,0:3,:,:],gradient_genimg_h_sobel.data[:,0:3,:,:]], dim=0), 'gradient_%s.png' % epoch, nrow=5, normalize=False)

        #targetimgs_blur2 = masks * targetimgs + torch.abs(1 - masks) * targetimgs_blur
        loss1 = self.criterionMSE(genimgs, targetimgs) 
        #loss1 = self.criterionMSE(genimgs[:,0:3,:,:], targetimgs[:,0:3,:,:]) + 0.5*self.criterionMSE(genimgs[:,3:6,:,:], targetimgs[:,3:6,:,:]) +\
        #0.2*self.criterionMSE(genimgs[:,6:9,:,:], targetimgs[:,6:9,:,:]) + 0.1*self.criterionMSE(genimgs[:,9:12,:,:], targetimgs[:,9:12,:,:])
        #if epoch % 10 == 0:
        #    save_image(torch.cat([targetimg,targetimg_blur,targetimgs_blur2[:,0:3,:,:]], dim=2), 'blur_%s.png' % epoch, nrow=5, normalize=False)
        #P1 = torch.sum(masks[:, 0:3, :, :]) / (h_img * w_img * channels * batch_size)
        #interval = P1 / 4.0
        #P2 = P1 - interval
        #P3 = P2 - interval
        #P4 = P3 - interval


        #compute 2rd loss
        relative_gradient1 = torch.norm(gradient_genimg[:, 0:3, : ,: ] / (gradient_targetimg[:, 0:3, : ,: ] * masks[:, 0:3, :, :] + self.e)) / (channels * h_img * w_img) 

        relative_gradient2 = torch.norm(gradient_genimg[:, 3:6, : ,: ] / (gradient_targetimg[:, 3:6, : ,: ] * masks[:, 3:6, :, :] + self.e)) / (channels * h_img * w_img)

        relative_gradient3 = torch.norm(gradient_genimg[:, 6:9, : ,: ] / (gradient_targetimg[:, 6:9, : ,: ] * masks[:, 6:9, :, :] + self.e)) / (channels * h_img * w_img)

        relative_gradient4 = torch.norm(gradient_genimg[:, 9:12, : ,: ] / (gradient_targetimg[:, 9:12, : ,: ] * masks[:, 9:12, :, :] + self.e)) / (channels * h_img * w_img)
        
        #loss2_1 = torch.abs(relative_gradient1)
        #loss2_2 = torch.abs(relative_gradient2)

        #loss2_3 = torch.abs(relative_gradient3)
        #loss2_4 = torch.abs(relative_gradient4)
        loss2 = relative_gradient1 + relative_gradient2 + relative_gradient3 + relative_gradient4 

        #masks_gradient_1 = torch.sum(masks[:, 0:3, :, :]) / (h_img * w_img * channels * batch_size)

        #masks_gradient_2 = torch.sum(masks[:, 3:6, :, :]) / (h_img * w_img * channels * batch_size)

        #masks_gradient_3 = torch.sum(masks[:, 6:9, :, :]) / (h_img * w_img * channels * batch_size)

        #masks_gradient_4 = torch.sum(masks[:, 9:12, :, :]) / (h_img * w_img * channels * batch_size)

        #P1 = masks_gradient_1
        #interval = P1 / 4.0  
        #P2 = P1 - interval
        #P3 = P2 - interval
        #P4 = P3 - interval



        #loss2_1 = torch.abs(masks_gradient_1 - self.P1)**2
        #loss2_2 = torch.abs(masks_gradient_2 - self.P2)**2
        #loss2_3 = torch.abs(masks_gradient_3 - self.P3)**2
        #loss2_4 = torch.abs(masks_gradient_4 - self.P4)**2
        #loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4

        #compute 3th loss


        #edge_detection1 = F.conv2d(genimgs, self.weight1, stride=1,padding=1).detach() 
        #edge_detection2 = F.conv2d(genimgs, self.weight2, stride=1,padding=1).detach() 
        
        #edge_detection = (torch.abs(edge_detection1) + torch.abs(edge_detection2)) / 2
        #edge_detection_threshold = torch.where(edge_detection > 2.5,torch.ones(edge_detection.shape).cuda(),torch.zeros(edge_detection.shape).cuda())


        #loss3 = torch.mean(self.criterionCLS(masks, GT_masks))


        #mutual_grdient = torch.abs(gradient_genimg - gradient_targetimg) * gradient_genimg


        #mutual_grdient_w = torch.abs(gradienth_genimg_w - gradienth_targetimg_w) * gradienth_genimg_w


        #loss3 = (torch.mean(mutual_grdient_h).type(torch.cuda.FloatTensor) + torch.mean(mutual_grdient_w).type(torch.cuda.FloatTensor)) / 2

        #relative_h = torch.abs(gradient_genimg_h) / (torch.abs(gradient_targetimg_h.detach()) + self.e)

        #relative_w = torch.abs(gradienth_genimg_w) / (torch.abs(gradienth_targetimg_w.detach()) + self.e)

        #loss2 = torch.mean(relative_h) + torch.mean(relative_w)
        #loss3 = torch.mean(mutual_grdient)
        #loss3 = (torch.mean(torch.exp(mutual_grdient_h)-1) + torch.mean(torch.exp(mutual_grdient_w)-1)) / 2




        # compute 4th loss


        #relative_h = torch.abs(gradient_genimg_h) / (torch.abs(gradient_maskimg_h.detach()) + self.e)

        #relative_w = torch.abs(gradienth_genimg_w) / (torch.abs(gradienth_maskimg_w.detach()) + self.e)

        #relative_h = torch.abs(gradient_genimg_h)
        #relative_w = torch.abs(gradienth_genimg_w)

        #masked_relative_h = torch.abs(1 - masks) * gradient_genimg_h 


        #masked_relative_w = torch.abs(1 - masks) * gradienth_genimg_w


        #masked_relative_keep = 5 * masks * torch.abs(gradient_genimg - gradient_targetimg) 

        #masked_relative_smooth = torch.abs(1 - masks) * gradient_genimg
        



        #upper = torch.where(gradient_targetimg_h > 0.5, torch.ones(gradient_targetimg_h.shape).cuda(), torch.zeros(gradient_targetimg_h.shape).cuda()) 
        #lower = torch.where(gradient_targetimg_h < 0.5, torch.ones(gradient_targetimg_h.shape).cuda(), torch.zeros(gradient_targetimg_h.shape).cuda())
        #print("sum upper: {} , sum lower: {}".format(torch.sum(upper),torch.sum(lower)))

        #masked_relative_h1 = masks[:,0:3,:,:] * torch.abs(gradient_genimg_h[:,0:3,:,:] - gradient_targetimg_h[:,0:3,:,:]) + torch.abs(1 - masks[:,0:3,:,:]) * relative_h[:,0:3,:,:] 

        #masked_relative_w1 = masks[:,0:3,:,:] * torch.abs(gradienth_genimg_w[:,0:3,:,:] - gradienth_targetimg_w[:,0:3,:,:]) + torch.abs(1 - masks[:,0:3,:,:]) * relative_w[:,0:3,:,:] 


        #masked_relative_h2 = masks[:,3:6,:,:] * torch.abs(gradient_genimg_h[:,3:6,:,:] - gradient_targetimg_h[:,3:6,:,:]) + torch.abs(1 - masks[:,3:6,:,:]) * relative_h[:,3:6,:,:]

        #masked_relative_w2 = masks[:,3:6,:,:] * torch.abs(gradienth_genimg_w[:,3:6,:,:] - gradienth_targetimg_w[:,3:6,:,:]) + torch.abs(1 - masks[:,3:6,:,:]) * relative_w[:,3:6,:,:]

        #masked_relative_h3 = masks[:,6:9,:,:] * torch.abs(gradient_genimg_h[:,6:9,:,:] - gradient_targetimg_h[:,6:9,:,:]) + torch.abs(1 - masks[:,6:9,:,:]) * relative_h[:,6:9,:,:]

        #masked_relative_w3 = masks[:,6:9,:,:] * torch.abs(gradienth_genimg_w[:,6:9,:,:] - gradienth_targetimg_w[:,6:9,:,:]) + torch.abs(1 - masks[:,6:9,:,:]) * relative_w[:,6:9,:,:]


        #masked_relative_h4 = masks[:,9:12,:,:] * torch.abs(gradient_genimg_h[:,9:12,:,:] - gradient_targetimg_h[:,9:12,:,:]) + torch.abs(1 - masks[:,9:12,:,:]) * relative_h[:,9:12,:,:]

        #masked_relative_w4 = masks[:,9:12,:,:] * torch.abs(gradienth_genimg_w[:,9:12,:,:] - gradienth_targetimg_w[:,9:12,:,:]) + torch.abs(1 - masks[:,9:12,:,:]) * relative_w[:,9:12,:,:]

        #masked_relative_h = 0.05 * masked_relative_h1 + 0.1 * masked_relative_h2 + 0.2 * masked_relative_h3 + 0.7 * masked_relative_h4

        #masked_relative_w = 0.05 * masked_relative_w1 + 0.1 * masked_relative_w2 + 0.2 * masked_relative_w3 + 0.7 * masked_relative_w4


        loss4 = self.criterionMSE(masks * canny * gradient_genimg, masks * canny * gradient_targetimg)





        #compute total loss
        totalloss = loss1 + self.u*loss2 + self.a*loss4
        #totalloss = self.a*loss4 
 
        print("term:{}  ,  epoch:{}  ,  loss1:{}  ,  loss2:{}  ,  loss4:{}  ,  totalloss:{}".format('smooth',epoch,loss1,loss2,loss4,totalloss))


        return totalloss

"""
class Mask_Loss(nn.Module):
    def __init__(self,cnn=vgg):
        super(Mask_Loss, self).__init__()
        self.criterionMSE = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionCLS = nn.BCELoss()
        self.a = 1.5

        self.u = 20

        self.b = 1.5
        self.P1 = 0.5
        self.P2 = 0.2
        self.P3 = 0.05
        self.P4 = 0.01
        self.e = 0.005


        #self.sobel_kernel1 = torch.cuda.FloatTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).expand(12,12,3,3)
        #self.sobel_kernel2 = torch.cuda.FloatTensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).expand(12,12,3,3)
        #self.weight1 = nn.Parameter(data=self.sobel_kernel1, requires_grad=False)
        #self.weight2 = nn.Parameter(data=self.sobel_kernel2, requires_grad=False)

    def forward(self,genimgs, targetimg, masks, epoch):
        #u = self.u * math.log(epoch + 2)
        # a = min(self.a * math.log(epoch + 2),0.04)
        targetimgs = torch.cat([targetimg, targetimg, targetimg, targetimg], dim=1)

        batch_size = genimgs.size()[0]

        channels = genimgs.size()[1]
        h_img = genimgs.size()[2]
        w_img = genimgs.size()[3]

        # compute struct image gradient
        gradient_genimg_h = F.pad(torch.abs(genimgs[:, :, 1:, :] - genimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_genimg_w = F.pad(torch.abs(genimgs[:, :, :, 1:] - genimgs[:, :, :, :w_img - 1]), (1, 0, 0, 0))
        # compute origion image gradient
        gradient_targetimg_h = F.pad(torch.abs(targetimgs[:, :, 1:, :] - targetimgs[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_targetimg_w = F.pad(torch.abs(targetimgs[:, :, :, 1:] - targetimgs[:, :, :, :w_img - 1]),(1, 0, 0, 0))
        # compute texture image gradient
        gradient_maskimg_h = F.pad(torch.abs(masks[:, :, 1:, :] - masks[:, :, :h_img - 1, :]), (0, 0, 1, 0))
        gradienth_maskimg_w = F.pad(torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :w_img - 1]), (1, 0, 0, 0))





        
        relative_h = torch.abs(gradient_genimg_h) / (torch.abs(gradient_targetimg_h) + self.e)
        relative_w = torch.abs(gradienth_genimg_w) / (torch.abs(gradienth_targetimg_w) + self.e)

        #upper = torch.where(gradient_targetimg_h > 0.5, torch.ones(gradient_targetimg_h.shape).cuda(), torch.zeros(gradient_targetimg_h.shape).cuda())
        #lower = torch.where(gradient_targetimg_h < 0.5, torch.ones(gradient_targetimg_h.shape).cuda(), torch.zeros(gradient_targetimg_h.shape).cuda())
        #print("sum upper: {} , sum lower: {}".format(torch.sum(upper),torch.sum(lower)))

        masked_relative_h = masks * torch.abs(gradient_genimg_h - 1) + 15 * torch.abs(1 - masks) * gradient_genimg_h

        masked_relative_w = masks * torch.abs(gradienth_genimg_w - 1) + 15 * torch.abs(1 - masks) * gradienth_genimg_w

        loss4 = (torch.mean(masked_relative_h) + torch.mean(masked_relative_w)) / 2
        totalloss = self.a*loss4
        print("term:{}  ,  epoch:{}  ,  loss4:{}  ,  totalloss:{}".format('mask',epoch,loss4,totalloss))  
        return totallos"""
