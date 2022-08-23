# FedDFKD
This repository is for summer intern's project: Communication-Efficient Federated Learning with heterogenous clients with Data-Free Knowledge Distillation

#### Code Instructions: 
#### Environment 
Python3.6  

We used pipreqs to generate the requirements.txt, thus we have the minimal packages needed.  

#### Code structure 
* main.py //For training the model 
* models.py //Our VAEs model for FMNIST, CIFAR10/100
* sampling.py // functions that generate non-iid datasets for federated learning
* utils.py // define functions that compute accuracy, soft prediction and model averaging
* mem_utils.py // Library for monitoring memory usage and training time
* option.py // define hyper-parameters
* Server/*.py // object definition for server in each method
* Client/*.py // object definition for client in each method

#### Parameters
* --dataset: 'CIFAR10', 'CIFAR100', ' SVHN', 'EMNIST'
* --batch_size: 64 by defalut 
* --num_epochs: number of global rounds, 50 by defalut
* --lr: learning rate, 0.001 by defalut
* --lr_sh_rate: period of learning rate decay, 10 by defalut
* --dropout_rate: drop out rate for each layer, 0.2 by defalut
* --clip_grad: maximum norm for gradient
* --num_users: number of clients, 10 by defalut
* --sampling_rate: proportion of clients send updates per round, 1 by defalut
* --local_ep: local epoch, 5 by defalut
* --beta: concentration parameter for Dirichlet distribution: 0.5 by defalut
* --seed: random seed(for better reproducting experiments): 0 by defalut
* --std: standard deviation by Differential Noise, 2 by defalut
* --code_len: length of latent vector, 32 by defalut
* --alg: 'FedAvg, FedProx, Moon, FedMD, Fedproto, FedDFKD'
* --eval_only: only ouput the testing accuracy during training and the running time
* --part: percentage of each local data
* --temp: temperture for soft prediction
* --lam: hyper-parameter for loss2
* --gamma: hyper-parameter for loss3
* --model: CNN resnet18 shufflenet
* --upload_model: allow clients to upload models to the server

#### Running the code for training and evaluation
We mainly use a .sh files to execute multiple expriements in parallel. 
The exprimenets are saved in checkpoint with unique id. Also, when the dataset is downloaded for the first time it takes a while. 


