python train.py --dataset ohsumed --lr 0.02 --dropout_rate_feat 0.4  -usewhole --dropout_rate 0 --dp1 0.1 --dp2 0.1 --epochs 400 --seed 2 
python train.py --dataset R52 --lr 0.02 -usewhole  --dp1 0.1 --dp2 0.1 --low_fre_word 0 --epochs 400 
python train.py --dataset R8 --lr 0.01 -usewhole --dp1 0.8 --dp2 0.2 --epochs 600 
python train.py --dataset 20ng --lr 0.02 -usewhole --device cuda:2 --dp1 0.3 --dp2 0.1 --seed 2 --epochs 1000 --seed 2