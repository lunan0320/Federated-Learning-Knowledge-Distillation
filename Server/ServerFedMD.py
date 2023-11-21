
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedMD import ClientFedMD
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc

class ServerFedMD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test, pub_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
        dict_pub = [np.random.randint(low=0,high=10000,size = 1000)]
        self.public_data = LocalDataloaders(pub_test,dict_pub,args.batch_size,ShuffleorNot = False,frac=1)[0]
    
    def Create_Clints(self):

        
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedMD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], loader_pub = self.public_data, idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.num_epochs)):
            Knowledges = []
            test_accuracy = 0
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                if epoch < 1:        
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    
                else:
                    w, loss = self.LocalModels[idx].update_weights_MD(global_round=epoch, knowledges = global_soft_prediciton, lam = 0.1, temp = self.args.temp)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    
                knowledges = self.LocalModels[idx].generate_knowledge(temp=self.args.temp)
                Knowledges.append(torch.stack(knowledges))
            global_soft_prediciton = []
            batch_pub = Knowledges[0].shape[0]
            for i in range(batch_pub):
                num = Knowledges[0].shape[1]
                soft_label = torch.zeros(num,self.args.num_classes)
                for idx in idxs_users:
                    soft_label += Knowledges[idx][i]
                soft_label = soft_label/ len(idxs_users)
                global_soft_prediciton.append(soft_label)
            del Knowledges
            gc.collect()

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