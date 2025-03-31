import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tsfm
import cv2
import os
from PIL import Image
from torchvision.io import read_image 
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)
import sys 
sys.path.append('/media/Datacenter_storage/')


class ImageDataset_Mayo_test(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        # Define the indices of the columns you want to extract
        label_columns_indices = [-1,-1,-1,-1]  # MACE_6mo, MACE_1yr, MACE_2yr, MACE_5yr
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[i] for i in label_columns_indices]
            for line in f:
                fields = line.strip('\n').split(',')
                image_path = fields[3]
                labels = [fields[i] for i in label_columns_indices]
               # bias = [fields[i] for i in bias_columns_indices]
                self._image_paths.append(image_path)
                self._labels.append(labels)
              #  self._bias.append(bias)
                
            
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image
    
    current_directory = os.getcwd()

    print("The script is running from:", current_directory)


    
    def __getitem__(self, idx):
        #print(f"image_path: {self._image_paths[idx]}")
        image_path = self._image_paths[idx]
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the current image in grayscale
        # im_gray = cv2.resize(image, (512, 512), interpolation = cv2.INTER_LANCZOS4)
        # image = (im_gray - np.min(im_gray)) / (np.max(im_gray) - np.min(im_gray))
        # image = np.stack([im_gray] * 3, axis=-1)
        # image = np.transpose(image, (2, 0, 1)) 
        # image = torch.tensor(image, dtype=torch.float32)

        image = cv2.imread(image_path)
        img_transform = tsfm.Compose([
        tsfm.ToTensor(),
        tsfm.Resize((512, 512), antialias=True),])
        image = img_transform(image)
        image = image.float()

        # Normalize to [0, 1]
        max_v = image.max()
        min_v = image.min()
        image = (image - min_v) * (1 / (max_v - min_v))

        # Rescale to the range [-2, 2]
        image = image * 4 - 2  # Maps [0, 1] to [-2, 2]
               
               

        labels = np.array(self._labels[idx]).astype(np.float32)
        path = self._image_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return image, labels
        elif self._mode in ['test', 'heatmap']:
            return image, path, labels
        else:
            raise Exception(f'Unknown mode: {self._mode}')

        
        



