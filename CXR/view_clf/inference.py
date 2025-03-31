import argparse
import torch
from DataLoader import OSADataset, transforms
from utils.data_model import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import torch, torch.nn as nn, torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from utils import utils
from tqdm.auto import tqdm
import pickle

# python inference.py

class Test:
    def __init__(self, test_loader, model, device):
        self.test_loader = test_loader
        self.model = model
        self.device = device

    def test(self):
        best_path = os.path.join("./view_clf/best.pth.tar") #enter your model path

        weights = self.model.state_dict()

        load_weights = list(torch.load(best_path, map_location=torch.device('cpu'))['model_state'].items())
        i=0
        for k, _ in weights.items():
            weights[k] = load_weights[i][1]
            i += 1

        self.model.load_state_dict(weights)
        self.model.eval().to(self.device)
        softmax = nn.Softmax()
        
        y_pred = []
        y_score = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.argmax(output, dim=1).cpu().detach().numpy().astype(int)
                y_pred += list(preds)
                # add a softmax layer to get probabilities
                scores = softmax(output)
                score = scores.squeeze()[preds].cpu().detach().numpy()
                y_score += list(score)
                      
            return y_pred, y_score


def view_clf_inference(inference_df):
    batch_size = 1
    num_workers = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    datagen = OSADataset(df=inference_df, transform=transforms, mode="test")
    dataloaders = torch.utils.data.DataLoader(dataset=datagen, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    tester = Test(
        test_loader=dataloaders,
        model=Model('Resnet18'),
        device=device,
        # enter your model path
        )
    y_pred, y_score = tester.test()
    #print(outputs)
    
    inference_df['view_pred'] = y_pred  # labels: AP = 0, lateral = 1
    inference_df['pred_score'] = y_score
    
    return inference_df
    

if __name__ == "__main__":
    # Example usage
    # Prepare your 'inference_df' DataFrame here
    inference_df = pd.read_csv("/MOSCARD/examples/mimic_test.csv")

    # Assuming 'transforms' is defined appropriately in the DataLoader module or elsewhere
    transformed_inference_df = view_clf_inference(inference_df)
    #print(transformed_inference_df)
    transformed_inference_df.to_csv("/MOSCARD/examples/mimic_test_view.csv")