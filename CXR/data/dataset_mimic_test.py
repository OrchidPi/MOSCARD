import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision.io import read_image 
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)
import sys 
sys.path.append('/media/Datacenter_storage/')


class ImageDataset_test(Dataset):
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
                image_path = fields[1]
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
        image = cv2.imread(image_path, 0)  # Load the current image in grayscale

        
        image = Image.fromarray(image)
        
        # Apply transformations if in 'train' mode
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        
        # Convert the PIL Image back to a NumPy array for further processing if necessary
        image = np.array(image)
        
        # Apply any additional transformations or processing
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)
       # bias = np.array(self._bias[idx]).astype(np.float32)
        path = self._image_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return image, labels
        elif self._mode in ['test', 'heatmap']:
            return image, path, labels
        else:
            raise Exception(f'Unknown mode: {self._mode}')

        
        



