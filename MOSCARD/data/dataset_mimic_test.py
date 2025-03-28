import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision.io import read_image 
from data.imgaug import GetTransforms
from data.utils import transform
from torchvision import transforms as tsfm
np.random.seed(0)
import sys 


class ImageDataset_Mayo_bimodal(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image1_paths = []
        self._image2_paths = []
        self._patient = []
        self._index = []
        self._labels = []
        self._mode = mode
        # Define the indices of the columns you want to extract
        label_columns_indices = [-1,-1,-1,-1]  # MACE_6mo, MACE_1yr, MACE_2yr, MACE_5yr

      

        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[i] for i in label_columns_indices]
            for line in f:
                fields = line.strip('\n').split(',')
                index = fields[2]
                patient = fields[0]
                image1_path = remove_space(fields[1])
                image2_path = remove_space(fields[3])
        
                label = [fields[i] for i in label_columns_indices]
                self._index.append(index)
                self._patient.append(patient)
                self._image1_paths.append(image1_path)
                self._image2_paths.append(image2_path)
                self._labels.append(label)

                
            
        self._num_image1 = len(self._image1_paths)
        self._num_image2 = len(self._image2_paths)

    def __len__(self):
        return self._num_image1
    
    current_directory = os.getcwd()

    #print("The script is running from:", current_directory)


    
    def __getitem__(self, idx):
        #print(f"image_path: {self._image_paths[idx]}")
        image1_path = self._image1_paths[idx]
        #image1_path = image1_path.replace('/media/Datacenter_storage/ChestXray_cardiology/', '/app/pngs/')
        image1 = cv2.imread(image1_path, 0)  # Load the current image in grayscale

        image1 = Image.fromarray(image1)


        # Apply transformations if in 'train' mode
        if self._mode == 'train':
            image1 = GetTransforms(image1, type=self.cfg.use_transforms_type)
        
        # Convert the PIL Image back to a NumPy array for further processing if necessary
        image1 = np.array(image1)
        # Apply any additional transformations or processing
        image1 = transform(image1, self.cfg)
        #print(f"image3:{image.shape}")
    
        
        image2_path = self._image2_paths[idx]
        #image2_path = image2_path.replace('/media/Datacenter_storage/jialu/external_ecg_images/', '/app/ecgs/')
        image2 = cv2.imread(image2_path)
        img_transform = tsfm.Compose([tsfm.ToTensor(),tsfm.Resize((512, 512)),])
        image2 = img_transform(image2)
        image2 = image2.float()
        max_v = image2.max()
        min_v = image2.min()
        image2 = (image2 - min_v)*(1/(max_v - min_v))
        image2 = image2 * 4 - 2
               

        labels = np.array(self._labels[idx]).astype(np.float32)
        index = self._index[idx]
        patient = self._patient[idx]
        path1 = self._image1_paths[idx]
        path2 = self._image2_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return image1, image2, labels
        elif self._mode in ['test', 'heatmap']:
            return index, patient, image1, image2, path1, path2, labels
        else:
            raise Exception(f'Unknown mode: {self._mode}')

def remove_space(s):
    if ' ' == s[0]:
        return s[1:]
    else:
        return s

        
        



