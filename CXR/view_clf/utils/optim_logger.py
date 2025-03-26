import torch, torch.nn as nn, torch.nn.functional
import pandas as pd
import numpy as np
import sklearn.model_selection
import torchvision.models as models
import torch.optim as optim
import sklearn.metrics
import pickle as pkl
import matplotlib.pyplot as plt
import os
import utils
#from hyperparameters import hyperparameters
from data_model import Model


class FinetuningModel:
    '''
    model_name: The name of the pre-trained model to use.
    hyperparameters: A dictionary of hyperparameters that are used to configure the model training.
    do_train: A boolean flag indicating whether to train the model or not.
    n_iters: The number of training iterations to perform.
    batch_size: The size of the mini-batch to use during training.
    metric_name: The name of the metric to use for evaluation.
    maximize_metric: A boolean indicating whether to maximize or minimize the specified metric.
    save_dir: The directory where the model and its training logs will be saved.
    '''
    def __init__(self, model_name, hyperparameters, do_train, n_iters, batch_size, metric_name, maximize_metric, save_dir):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.train()
        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=hyperparameters["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hyperparameters["step_size"], gamma=0.11)
        self.save_dir = utils.get_save_dir(hyperparameters['save_dir'], training=True if do_train else False)
        self.logger = utils.get_logger(self.save_dir, "finetuning")
        self.logger.propagate = False
        self.saver = utils.CheckpointSaver(save_dir=self.save_dir, metric_name=metric_name,maximize_metric=maximize_metric, log=self.logger)