import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.root = root

        #self.files_edge = sorted(glob.glob(root + '/mask_Images/*.*'))
        self.files_img = sorted(os.listdir(self.root + '/Images/'))


    def __getitem__(self, index):


        filename = self.files_img[index % len(self.files_img)] 
        img_path = self.root + "/Images/" + filename
        blur_path1 = self.root + "/smooth_Images/" + filename
        blur_path2 = self.root + "/smooth_Images/" + filename
        blur_path3 = self.root + "/smooth_Images/" + filename
        blur_path4 = self.root + "/smooth_Images/" + filename

        edge_path1 = self.root + "/mask_Images/" + filename.split(".")[0] + "_0.png" 
        edge_path2 = self.root + "/mask_Images/" + filename.split(".")[0] + "_1.png"
        edge_path3 = self.root + "/mask_Images/" + filename.split(".")[0] + "_2.png"
        edge_path4 = self.root + "/mask_Images/" + filename.split(".")[0] + "_3.png"

        gradient_path1 = self.root + "/mask_Images2/" + filename.split(".")[0] + "_0.png"
        gradient_path2 = self.root + "/mask_Images2/" + filename.split(".")[0] + "_1.png"
        gradient_path3 = self.root + "/mask_Images2/" + filename.split(".")[0] + "_2.png"
        gradient_path4 = self.root + "/mask_Images2/" + filename.split(".")[0] + "_3.png"

        item_edge1 = self.transform(Image.open(edge_path1).convert("RGB"))
        item_edge2 = self.transform(Image.open(edge_path2).convert("RGB"))
        item_edge3 = self.transform(Image.open(edge_path3).convert("RGB"))
        item_edge4 = self.transform(Image.open(edge_path4).convert("RGB"))
        item_edge = torch.cat([item_edge1,item_edge2,item_edge3,item_edge4],dim=0)
        item_img = self.transform(Image.open(img_path).convert("RGB"))

        item_blur1 = self.transform(Image.open(blur_path1).convert("RGB"))
        item_blur2 = self.transform(Image.open(blur_path2).convert("RGB"))
        item_blur3 = self.transform(Image.open(blur_path3).convert("RGB"))
        item_blur4 = self.transform(Image.open(blur_path4).convert("RGB"))
        item_blur = torch.cat([item_blur1, item_blur2, item_blur3, item_blur4],dim=0)        

        item_gradient1 = self.transform(Image.open(gradient_path1).convert("RGB"))
        item_gradient2 = self.transform(Image.open(gradient_path2).convert("RGB"))
        item_gradient3 = self.transform(Image.open(gradient_path3).convert("RGB"))
        item_gradient4 = self.transform(Image.open(gradient_path4).convert("RGB"))
        item_gradient = torch.cat([item_gradient1, item_gradient2, item_gradient3, item_gradient4],dim=0)


        return {'edge':item_edge, 'img':item_img, 'blur':item_blur, 'gradient':item_gradient}


    def __len__(self):
        return len(self.files_img)
