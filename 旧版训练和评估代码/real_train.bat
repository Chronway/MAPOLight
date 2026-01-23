@echo off

python train.py -a PPO -n real -p 1

python train.py -a PPO -n real -p 0.2

python train.py -a PPO -n real -p 0.1

python train.py -a PPO -n real -p 0.05

python train.py -a A3C -n real -p 1

python train.py -a A3C -n real -p 0.2

python train.py -a A3C -n real -p 0.1

python train.py -a A3C -n real -p 0.05