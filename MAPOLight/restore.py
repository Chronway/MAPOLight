# 输入 trial 的保存路径，继续训练
# 使用 Ctrl+C 中断某次 trial 后，利用 Tuner.restore 方法，可以继续上次的进度训练
# example: python restore.py /data/lyq/cav_0.05/A3C
# 需要通过 register_env 注册实验需要的环境，否则恢复时会没有环境
# 需保证恢复时的环境与上次训练时的环境一致

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


current_dir = os.path.dirname(os.path.realpath(__file__))    # 获取当前文件夹

net_file = current_dir+'/sumo_files/4gridnet.net.xml'
rou_file = current_dir+'/sumo_files/4groutes.xml'
cfg_file = current_dir+'/sumo_files/4grid.sumocfg'

#current_directory = os.path.dirname(os.path.abspath(__file__))

my_env = MyPettingZooEnv(env(
    net_file=net_file,
    route_file=rou_file,
    config_file = cfg_file,
    PR=0.05,
    out_csv_name=None,
    use_gui=False,
    num_seconds=3500,
    #time_to_load_vehicles=10,
    begin_time=10, #原来的time_to_load_vehicles
    max_depart_delay=0,
    cav_env = True, # has cav env
    collaborate=True
))
# Register the model and environment
register_env("net", lambda _: my_env)

#ModelCatalog.register_custom_model("my_model", MyModel)

ray.init(num_cpus=4,num_gpus=1)

c = Tuner.restore(path=trial_path, trainable=trial_path.split('/')[-1])
c.fit()
