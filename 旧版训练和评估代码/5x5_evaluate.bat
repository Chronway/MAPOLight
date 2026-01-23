@echo off
echo "! 该脚本中的检查点路径需要根据实际情况配置 !"

python test2.py -a a3c -c "5x5_weights/A3C/1/checkpoint_019043" -p 1 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a a3c -c "5x5_weights/A3C/0.05/checkpoint_014438" -p 0.05 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a a3c -c "5x5_weights/A3C/0.1/checkpoint_016867" -p 0.1 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a a3c -c "5x5_weights/A3C/0.2/checkpoint_017343" -p 0.2 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width

python test2.py -a ppo -c "5x5_weights/PPO/1/checkpoint_002006" -p 1 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a ppo -c "5x5_weights/PPO/0.05/checkpoint_002115" -p 0.05 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a ppo -c "5x5_weights/PPO/0.1/checkpoint_000937" -p 0.1 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width
python test2.py -a ppo -c "5x5_weights/PPO/0.2/checkpoint_001804" -p 0.2 --co -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -o 5x5 -t 4800 --128width