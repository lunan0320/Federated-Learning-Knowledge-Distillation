CUDA_VISIBLE_DEVICES=7 python main.py --dataset 'SVHN' --batch_size 64  --num_epochs 50 --clip_grad 1.1 --lr 0.001 --num_clients 10 --num_classes 10 --sampling_rate 1 --local_ep 3 --beta 0.5 --seed 0 --code_len 50 --alg 'FedAvg' --part 0.1 --model 'resnet18'  --temp  0.5  

