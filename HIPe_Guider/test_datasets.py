import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.root = root
        self.files_img = sorted(os.listdir(self.root + '/Images/'))

    def __getitem__(self, index):

        filename = self.files_img[index % len(self.files_img)]
        img_path = self.root + '/Images/' + filename
        #item_edge = self.transform(Image.open(edge_path).convert("RGB"))
        item_img = self.transform(Image.open(img_path).convert("RGB"))

        return {'img':item_img, 'img_path': img_path, 'filename': filename}

    def __len__(self):
        return len(self.files_img)
