import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
import os
from Config import dataread
from hyperparameters import hyperparameters
import torch


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
        self.targets = df["Binary_Label"].values

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
        config = dataread(
            hyperparameters["data_dir"],
            hyperparameters["train_file"],
            hyperparameters["test_file"],
            hyperparameters["val_file"]
            )

        image_name = os.path.join(config.data_dir, self.df.at[index, "PNG_FIXED"])
        y = self.df.at[index, "Binary_Label"]

        one_hot = np.zeros(len(self.df["Binary_Label"].unique()))
        try:
            x = io.imread(image_name)
            x = (((x - x.min()) / x.max() - x.min())*255).astype(np.uint8)
            if self.mode == 'train':
                if isinstance(x, np.ndarray):
                    x = Image.fromarray(x)
                for aug in self.augmentations:
                    x = aug(x)
            x = self.transform(x)
            one_hot[y] = 1.
            one_hot = np.array(one_hot).astype(np.float32)

            return x, one_hot

        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return None