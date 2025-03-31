import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
import cv2
import os
from utils.Config import dataread
from utils.hyperparameters import hyperparameters
import torch
import pydicom
from data.imgaug import GetTransforms
from data.utils import transform


class OSADataset(Dataset):
    """
        Class for loading the images and their corresponding labels.
        Parameters:
        df (pandas.DataFrame): DataFrame with image information.
        transform (callable): Data augmentation method.
        mode (str): Mode of the data ('train', 'test', 'val').
    """

    def __init__(self, df, transform=None, mode=None, augmentations=None):
        self.df = df
        self.mode = mode

        self.augmentations = augmentations or ([
            transforms.RandomRotation(degrees=15),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47, 0.47, 0.47],std=[0.3033, 0.3033, 0.3033]),
            ])        

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.df)
    

    def __getitem__(self, index):
        """
        Get an image and its corresponding label.
        Parameters:
        index (int): Index of the image.
        Returns:
        tuple: (image, label)
        """
        #config = dataread(
        #    hyperparameters["data_dir"],
        #    hyperparameters["train_file"],
        #   hyperparameters["test_file"],
        #    hyperparameters["val_file"]
        #    )

        #image_name = os.path.join(config.data_dir, self.df.at[index, "PNG_FIXED"])
        image_name = self.df.at[index, "CXR_Path"].strip()
        #image_name = self.df.path.iloc[index] # absolute path to the image

        try:
            #x = io.imread(image_name)
            # different processing, straight from discoms
            x = cv2.imread(image_name, 0)
            # Save the inverted image
            x = Image.fromarray(x)
            x = GetTransforms(x, type='None')
            x = np.array(x)
            x = transform(x)

            
            if self.mode == 'train':
                if isinstance(x, np.ndarray):
                    x = Image.fromarray(x)
                for aug in self.augmentations:
                    x = aug(x)
            x = self.transform(x)
            return x

        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return None