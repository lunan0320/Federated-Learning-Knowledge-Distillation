import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    #Data specifc paremeters
    parser.add_argument('--dataset', default='CIFAR10',
                        help='CIFAR10, CIFAR100, SVHN, EMNIST') 
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--num_clients', default=10,
                        help='number of local models')
    
    parser.add_argument('--num_classes', default=10,
                        help='number of classes')
    
    parser.add_argument('--sampling_rate', default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep', default=5,
                        help='iterations of local updating')
    parser.add_argument('--beta', default=0.5,
                        help='beta for non-iid distribution')
    parser.add_argument('--seed', default=0,
                        help='random seed for generating datasets')
    parser.add_argument('--code_len', default=32,
                        help='length of code')
    parser.add_argument('--alg', default='FedAvg',
                        help='FedAvg, FedProx, Moon, FedMD, Fedproto, FedDFKD')
    
    parser.add_argument('--lam', type=float, default=0.1,
                        help='hyper-parameter for loss2')
    
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='hyper-parameter for loss3')
    
    parser.add_argument('--std', type=float, default=2,
                        help='std of gaussian noise ')
    
    parser.add_argument('--part', default=0.1,
                        help='percentage of each local data')
    
    parser.add_argument('--sample', default=1,
                        help='sample rate of clients')
    
    parser.add_argument('--temp', default=0.5,
                        help='temperture for soft prediction')
    
    parser.add_argument('--model', default= 'resnet18',
                        help='CNN resnet18 shufflenet')
    
    parser.add_argument('--upload_model', action="store_true", default= False,
                        help='allow clients to upload models to the server')
    
    parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')

    args = parser.parse_args('')
    return args

