import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        #self.files_edge = sorted(glob.glob(root + '/canny_edge/*.png'))
        self.root = root
        self.files_img = sorted(os.listdir(root + '/images/'))

    def __getitem__(self, index):

        filename = self.files_img[index % len(self.files_img)] 
        img_path = self.root + '/images/' + filename
        item_img = self.transform(Image.open(img_path).convert("RGB"))

        #filename_edge1 = filename.split(".")[0] + "_1.png"
        #filename_edge2 = filename.split(".")[0] + "_2.png"
        #filename_edge3 = filename.split(".")[0] + "_3.png"
        filename_edge = filename.split(".")[0] + ".png"
        
        edge_path = self.root + '/groundTruth_png/' + filename_edge
        item_edge = self.transform(Image.open(edge_path).convert("RGB"))
        

        return {'img':item_img, 'edge':item_edge}

    def __len__(self):
        return len(self.files_img)
