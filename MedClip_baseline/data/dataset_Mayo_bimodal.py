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
sys.path.append('/media/Datacenter_storage/')


class ImageDataset_Mayo_bimodal(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._bias_header = None
        self._image1_paths = []
        self._image2_paths = []
        self._image_paths_inverted = []
        self._index = []
        self._patient = []
        self._bias = []
        self._causal = []
        self._causal2 = []
        self._labels = []
        self._mode = mode
        # Define the indices of the columns you want to extract
        label_columns_indices = [9,10,11,12]  # MACE_6mo, MACE_1yr, MACE_2yr, MACE_5yr

        ##CKD and Chf
        causal_columns_indices = [5,6]
        causal2_columns_indices = [7,8]

        bias_columns_indices = [3,4]

        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[i] for i in label_columns_indices]
            self._bias_header = [header[i] for i in bias_columns_indices]
            self._causal_header = [header[i] for i in causal_columns_indices]
            self._causal2_header = [header[i] for i in causal2_columns_indices]
            for line in f:
                fields = line.strip('\n').split(',')
                index = fields[0]
                patient = fields[1]
                image1_path = remove_space(fields[2])
                image2_path = remove_space(fields[-2])
                print(f"image_path1: {image1_path}")
                print(f"image_path2: {image2_path}")
                if fields[-1] == 'MONOCHROME1':
                    image_inverted = remove_space(fields[2])
                    self._image_paths_inverted.append(image_inverted)
                
                #print(f"ckd:{fields[5]}")
                label = [fields[i] for i in label_columns_indices]
                causal = [fields[i] for i in causal_columns_indices]
                causal2 = [fields[i] for i in causal2_columns_indices]
                bias = [fields[i] for i in bias_columns_indices]
                self._index.append(index)
                self._patient.append(patient)
                self._image1_paths.append(image1_path)
                self._image2_paths.append(image2_path)
                self._labels.append(label)
                self._bias.append(bias)
                self._causal.append(causal)
                self._causal2.append(causal2)
                
            
        self._num_image1 = len(self._image1_paths)
        self._num_image2 = len(self._image2_paths)

    def __len__(self):
        return self._num_image1
    
    current_directory = os.getcwd()

    print("The script is running from:", current_directory)


    
    def __getitem__(self, idx):
        #print(f"image_path: {self._image_paths[idx]}")
        image1_path = self._image1_paths[idx]
        image1 = cv2.imread(image1_path, 0)  # Load the current image in grayscale

        # Check if this image needs to be inverted
        if image1_path in self._image_paths_inverted:
            # This image needs to be inverted
            inverted_image_array = 255 - image1
            image1 = inverted_image_array  # Now 'image' is the inverted image

        image1 = Image.fromarray(image1)


        # Apply transformations if in 'train' mode
        if self._mode == 'train':
            image1 = GetTransforms(image1, type=self.cfg.use_transforms_type)
        
        # Convert the PIL Image back to a NumPy array for further processing if necessary
        image1 = np.array(image1)
        # Apply any additional transformations or processing
        image1 = transform(image1, self.cfg)
        #print(f"image1: max={image1.max()}, min={image1.min()}")
        #print(f"image3:{image.shape}")
    
        
        image2_path = self._image2_paths[idx]
        # image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)  # Load the current image in grayscale
        # im_gray2 = cv2.resize(image2, (512, 512), interpolation = cv2.INTER_LANCZOS4)
        # image2 = (im_gray2 - np.min(im_gray2)) / (np.max(im_gray2) - np.min(im_gray2))
        # image2 = np.stack([im_gray2] * 3, axis=-1)
        # image2 = np.transpose(image2, (2, 0, 1)) 
        # image2 = torch.tensor(image2, dtype=torch.float32)
        image2 = cv2.imread(image2_path)
        img_transform = tsfm.Compose([tsfm.ToTensor(),tsfm.Resize((512, 512)),])
        image2 = img_transform(image2)
        image2 = image2.float()
        max_v = image2.max()
        min_v = image2.min()
        image2 = (image2 - min_v)*(1/(max_v - min_v))
        # Rescale to the range [-2, 2]
        image2 = image2 * 4 - 2  # Maps [0, 1] to [-2, 2]
        #image2 = image2.permute(1,2,0)
        #print(f"image2: max={image2.max()}, min={image2.min()}")
               

        labels = np.array(self._labels[idx]).astype(np.float32)
        bias = np.array(self._bias[idx]).astype(np.float32)
        causal = np.array(self._causal[idx]).astype(np.float32)
        causal2 = np.array(self._causal2[idx]).astype(np.float32)
        index = self._index[idx]
        patient = self._patient[idx]
        path1 = self._image1_paths[idx]
        path2 = self._image2_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return index, patient, image1, image2, labels, bias, causal
        elif self._mode in ['test', 'heatmap']:
            return index, patient, image1, image2, path1, path2, labels
        else:
            raise Exception(f'Unknown mode: {self._mode}')

def remove_space(s):
    if ' ' == s[0]:
        return s[1:]
    else:
        return s

        
        



