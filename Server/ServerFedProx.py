
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedProx import ClientFedProx
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
class ServerFedProx(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
       
    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedProx(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.num_epochs)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights_Prox(global_round=epoch, lam=0.1)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc


             # update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average local test accuracy:', test_accuracy / self.args.num_clients)
            print('global test accuracy: ', self.global_test_accuracy())
            
        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()