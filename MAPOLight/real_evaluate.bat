@echo off
echo "! The checkpoint path in this script needs to be configured according to your specific environment !"

python test2.py -a a3c -c "weights/A3C/1/checkpoint_020543" -p 1 --co
python test2.py -a a3c -c "weights/A3C/0.05/checkpoint_009628" -p 0.05 --co
python test2.py -a a3c -c "weights/A3C/0.1/checkpoint_022328" -p 0.1 --co
python test2.py -a a3c -c "weights/A3C/0.2/checkpoint_001878" -p 0.2 --co

python test2.py -a ppo -c "weights/PPO/1/checkpoint_001038" -p 1 --co
python test2.py -a ppo -c "weights/PPO/0.05/checkpoint_002253" -p 0.05 --co
python test2.py -a ppo -c "weights/PPO/0.1/checkpoint_002597" -p 0.1 --co
python test2.py -a ppo -c "weights/PPO/0.2/checkpoint_001018" -p 0.2 --co
