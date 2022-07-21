python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedAvg' --part 0.2 --model 'resnet18' --upload_model > ./training_logs/cifar_fedavg_alg2_seed0.out

python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedProx' --part 0.2 --model 'resnet18' --upload_model > ./training_logs/cifar_fedprox_alg2_seed0.out

python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedMD' --part 0.2 --model 'resnet18'  --temp 0.5 --upload_model > ./training_logs/cifar_fedmd_alg2_seed0.out

python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedProto' --part 0.2 --model 'resnet18' --gamma 0.1 --upload_model > ./training_logs/cifar_fedproto_alg2_seed0.out

python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedDFKD' --part 0.2 --model 'resnet18' --lam 0.1 --gamma 0 --std 2 --upload_model > ./training_logs/cifar_feddfkd_alg2_seed0.out

python main.py --dataset 'CIFAR10' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 20 --num_classes 10 --sampling_rate 1 --local_ep 5 --beta 0.5 --seed 0 --code_len 64 --alg 'FedDFKD' --part 0.2 --model 'resnet18' --lam 0.1 --gamma 0.1 --std 2 --upload_model > ./training_logs/cifar_fedcombine_alg2_seed0.out