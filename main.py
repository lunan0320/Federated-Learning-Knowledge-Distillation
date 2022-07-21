
import torch
import numpy as np
import os,sys,os.path
from tensorboardX import SummaryWriter
import pickle
from torch import nn
import hashlib



# packages from CityChan
from models import CNNFemnist,ResNet18,ShuffLeNet
from sampling import LocalDataset, LocalDataloaders, partition_data
from option import args_parser

from Server.ServerFedAvg import ServerFedAvg
from Server.ServerFedProx import ServerFedProx
from Server.ServerFedMD import ServerFedMD 
from Server.ServerFedProto import ServerFedProto
from Server.ServerFedDFKD import ServerFedDFKD



torch.set_default_dtype(torch.float64)
print(torch.__version__)
torch.cuda.is_available()
device = torch.device("cuda:0")
args = args_parser()
np.set_printoptions(threshold=np.inf)
# change available gpu number 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

print(args)
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only':
        continue
    args_hash += str(k)+str(v)
    
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()





train_dataset,testset, dict_users, dict_users_test = partition_data(n_users = args.num_clients, alpha=args.beta,rand_seed = args.seed, dataset=str(args.dataset))




Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=args.part)
Loaders_test = LocalDataloaders(testset,dict_users_test,args.batch_size,ShuffleorNot = True,frac=args.part)
global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, num_workers=2)

for idx in range(args.num_clients):
    counts = [0]*args.num_classes
    for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print('Client {} data distribution:'.format(idx))
    print(counts)





logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Checkpoint dir:', checkpoint_dir)





if args.model == 'CNN':
    global_model = CNNFemnist(args, code_length=args.code_len, num_classes = args.num_classes)
    
if args.model == 'resnet18':
    global_model = ResNet18(args, code_length=args.code_len, num_classes = args.num_classes)

if args.model == 'shufflenet':  
    global_model = ShuffLeNet(args, code_length=args.code_len, num_classes = args.num_classes)

   
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
global_model = nn.DataParallel(global_model)
global_model.to(device)





if args.alg == 'FedAvg':
    server = ServerFedAvg(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedProx':
    server = ServerFedProx(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedMD':
    server = ServerFedMD(args,global_model,Loaders_train,Loaders_test,global_loader_test,testset,logger,device)
if args.alg == 'FedProto':    
    server = ServerFedProto(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedDFKD':    
    server = ServerFedDFKD(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)


server.Create_Clints()
server.train()



server.global_test_accuracy()
save_path = checkpoint_dir + args_hash + '.pth'
if args.upload_model == True:
    server.Save_CheckPoint(save_path)
    print('Model is saved on: ')
    print(save_path)






