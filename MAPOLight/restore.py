# restore an interrupted trial
# example: python restore.py <trial path>
# must keep the env name same
# must check whether `net_file`,`rou_file`,`cfg_file` is correct

from ray.tune import Tuner
import sys
from ray.tune.registry import register_env
import os
from utils.mypettingzoo import MyPettingZooEnv
from env.SignalEnv import env
import ray
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


trial_path = sys.argv[1]
print(f"path: {trial_path}")


current_dir = os.path.dirname(os.path.realpath(__file__))

net_file = current_dir+'/sumo_files/4gridnet.net.xml'
rou_file = current_dir+'/sumo_files/4groutes.xml'
cfg_file = current_dir+'/sumo_files/4grid.sumocfg'

my_env = MyPettingZooEnv(env(
    net_file=net_file,
    route_file=rou_file,
    config_file = cfg_file,
    PR=0.05,
    out_csv_name=None,
    use_gui=False,
    num_seconds=3500,
    begin_time=10,
    max_depart_delay=0,
    cav_env = True, # has cav env
    collaborate=True
))
# Register the model and environment
register_env("net", lambda _: my_env)

ray.init(num_cpus=4,num_gpus=1)

c = Tuner.restore(path=trial_path, trainable=trial_path.split('/')[-1])
c.fit()
