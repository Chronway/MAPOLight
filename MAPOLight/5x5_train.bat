@echo off

python train.py -a PPO -n 5x5 -p 1

python train.py -a PPO -n 5x5 -p 0.2

python train.py -a PPO -n 5x5 -p 0.1

python train.py -a PPO -n 5x5 -p 0.05

python train.py -a A3C -n 5x5 -p 1

python train.py -a A3C -n 5x5 -p 0.2

python train.py -a A3C -n 5x5 -p 0.1

python train.py -a A3C -n 5x5 -p 0.05