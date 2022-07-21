
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
from Client.ClientBase import Client
import gc
class ClientFedMD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, Loader_train,loader_test, loader_pub,idx, logger, code_length, num_classes, device):
        super().__init__(args, model, Loader_train,loader_test,idx, logger, code_length, num_classes, device)
        self.loader_pub = loader_pub
        
    def update_weights(self,global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                p = self.model(X).double()
                loss = self.ce(p,y)               
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.idx, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def update_weights_MD(self,knowledges, lam, temp, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        global_soft_prediction =  torch.stack(knowledges)
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                Z = self.model(X).double()
                loss1 = self.ce(Z,y)
                loss2 = torch.tensor(0.0).to(self.device)
                for idx, (X_pub,y_pub) in enumerate(self.loader_pub):
                    if idx == batch_idx:
                        X_pub = X_pub.to(self.device)
                        y_pub = y_pub.to(self.device)
                        Z_pub = self.model(X_pub).double()
                        Q_pub = soft_predict(Z_pub,temp).to(self.device)
                        loss2 -= self.kld(Q_pub,global_soft_prediction[idx].to(self.device))
                
                loss = loss1 + lam*loss2
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f} '.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss1.item(),loss2.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()
        num_classes = self.model.module.num_classes
        soft_predictions = []
        for batch_idx, (X, y) in enumerate(self.loader_pub):
            X = X.to(self.device)
            y = y
            Z = self.model(X) 
            Q = soft_predict(Z,temp).to(self.device).detach().cpu()
            soft_predictions.append(Q)
            del X
            del y
            del Z
            del Q
            gc.collect()
         
        return soft_predictions