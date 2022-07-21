
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict

class Client(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model,Loader_train,loader_test,idx, logger, code_length, num_classes, device):
        self.args = args
        self.logger = logger
        self.trainloader = Loader_train
        self.testloader = loader_test
        self.idx = idx
        self.ce = nn.CrossEntropyLoss() 
        self.device = device
        self.code_length = code_length
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)
   
    
    def test_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            p = self.model(X).double()
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt

    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)