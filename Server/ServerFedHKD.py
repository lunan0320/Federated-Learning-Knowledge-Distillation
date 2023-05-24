
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedHKD import ClientFedHKD
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc

class ServerFedHKD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)

    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedHKD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
    def global_knowledge_aggregation(self, features,soft_prediction):
        global_local_features = dict()
        global_local_soft_prediction = dict()
        for [label, features] in features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                global_local_features[label] = [feature / len(features)]
            else:
                global_local_features[label] = [features[0].data]

        for [label, soft_prediction] in soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data
                for i in soft_prediction:
                    soft += i.data
                global_local_soft_prediction[label] = [soft / len(soft_prediction)]
            else:
                global_local_soft_prediction[label] = [soft_prediction[0].data]

        return global_local_features,global_local_soft_prediction

    def train(self):
        global_features = {}
        global_soft_prediction = {}
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
                    w, loss = self.LocalModels[idx].update_weights_HKD(global_round=epoch, global_features=global_features, global_soft_prediction=global_soft_prediction, lam = self.args.lam, gamma = self.args.gamma, temp = self.args.temp)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    
                local_features,local_soft_predictions  = self.LocalModels[idx].generate_knowledge(temp = self.args.temp)
                global_features.update(local_features)
                global_soft_prediction.update(local_soft_predictions)
                del local_features
                del local_soft_predictions
                gc.collect()
 

             # update global weights
            global_weights = average_weights(local_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average local test accuracy:', test_accuracy / self.args.num_clients)
            print('global test accuracy: ', self.global_test_accuracy())
            
        print('Training is completed.')
        self.global_model.load_state_dict(global_weights)
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()
