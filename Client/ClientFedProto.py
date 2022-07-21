
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
class ClientFedProto(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, Loader_train,loader_test,idx, logger, code_length, num_classes, device):
        super().__init__(args, model, Loader_train,loader_test,idx, logger, code_length, num_classes, device)
    
    
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

    
    def update_weights_Proto(self,global_features, gamma, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        tensor_global_features = self.dict_to_tensor(global_features).to(self.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                F = self.model.module.feature_extractor(X)
                Z = self.model(X).double()
                loss1 = self.ce(Z,y)
                target_features = copy.deepcopy(F.data)

                
                for i in range(y.shape[0]):
                    if int(y[i]) in global_features.keys():
                        target_features[i] = global_features[int(y[i])][0].data
    
                        
                target_features = target_features.to(self.device)
                if len(global_features) == 0:
                    loss2 = 0*loss1
                else:
                    loss2 = self.mse(F,target_features)
                loss = loss1 + gamma*loss2
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm =1.1)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f}  '.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss1.item(),loss2.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
                        
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    # generate knowledge for FedDFKD
    def generate_knowledge(self):
        self.model.to(self.device)
        self.model.eval()        
        local_features = {}
        local_soft_prediction = {}
        num_classes = self.model.module.num_classes
        features = [torch.zeros(self.code_length).to(self.device)]*num_classes
        soft_predictions = [torch.zeros(num_classes).to(self.device)]*num_classes
        count = [0]*num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y
            F = self.model.module.feature_extractor(X)
            Z = self.model(X) 
            Q = soft_predict(Z,self.args.temp).to(self.device)
            m = y.shape[0]
            for i in range(len(y)):
                if y[i].item() in local_features:
                    local_features[y[i].item()].append(F[i,:])
                    local_soft_prediction[y[i].item()].append(Q[i,:])
                else:
                    local_features[y[i].item()] = [F[i,:]]
                    local_soft_prediction[y[i].item()]  = [Q[i,:]] 
            del X
            del y
            del F
            del Z
            del Q
            gc.collect()
            
        features = self.local_knowledge_aggregation(local_features)
        
        return features
    
    def local_knowledge_aggregation(self,local_features):
        agg_local_features = dict()
        for [label, features] in local_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data   
                agg_local_features[label] = [feature / len(features)]
            else:
                agg_local_features[label] = [features[0].data]
        
        return agg_local_features
    
    def dict_to_tensor(self, dic):
        lit = []
        for key,tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit