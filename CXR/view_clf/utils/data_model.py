import torch, torch.nn as nn, torch.nn.functional
import torchvision.models as models
import torch.optim as optim
import os
#import utils
#import hyperparameters

class Model(nn.Module):
    def __init__(self, pretrained=False):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax()
        )

        for m in self.model.modules():
            #kaiming weight initialization
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def freeze_layers(self, block):
        count = 0
        flg = 0
        for layer_name, param in reversed(list(self.model.named_parameters())):
            if flg == 1:
                if block in layer_name:
                    param.requires_grad = False
                    count += 1
            else:
                count += 1
                if block in layer_name:
                    flg = 1
    
    def forward(self, x): 
        x = self.model(x)
        return x


