import os
from typing import Literal

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["LIBSUMO_AS_TRACI"] = "1"

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.a3c.a3c import A3CConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.tune.registry import register_env
from env.SignalEnv import env
from utils.mypettingzoo import MyPettingZooEnv
import argparse
import logging

logger = logging.getLogger(__name__)

from ray.tune import CLIReporter
import ray.tune.stopper.stopper

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--arch", default=["PPO"], dest="arch", nargs='+')
parser.add_argument("-n", "--net", default=["real"], dest="net", nargs='+')
parser.add_argument("-p", "--pr", default=["1"], dest="pr", nargs='+')

def main(run, net, pr):
    ray.init(num_cpus=6, num_gpus=1)

    assert run in ["PPO", "A3C", "SAC", "DQN"]
    assert net in ["4x4", "6x6", "real", "5x5"]
    assert pr in ["0.05", "0.1", "0.2", "1"]

    if pr == "1":
        enable_cv = False
    else:
        enable_cv = True

    current_dir = os.path.dirname(os.path.realpath(__file__))

    if net == "4x4":
        net_file = current_dir + '/sumo_files/4gridnet.net.xml'
        rou_file = current_dir + '/sumo_files/4groutes.xml'
        cfg_file = current_dir + '/sumo_files/4grid.sumocfg'
        num_second = 3000
    elif net == "5x5":
        net_file = current_dir + '/sumo_files/5net.net.xml'
        rou_file = current_dir + '/sumo_files/5groutes.xml'
        cfg_file = current_dir + '/sumo_files/5grid.sumocfg'
        num_second = 3300
    elif net == "6x6":
        net_file = current_dir + '/sumo_files/6gridnet.net.xml'
        rou_file = current_dir + '/sumo_files/6groutes.xml'
        cfg_file = current_dir + '/sumo_files/6grid.sumocfg'
        num_second = 3000
    elif net == "real":
        net_file = current_dir + '/sumo_files/moco.net.xml'
        rou_file = current_dir + '/sumo_files/moco_jtr_out.rou.xml'
        cfg_file = current_dir + '/sumo_files/testmap.sumocfg'
        num_second = 7300

    my_env = MyPettingZooEnv(env(
        net_file=net_file,
        route_file=rou_file,
        config_file=cfg_file,
        PR=float(pr),
        out_csv_name=None,
        use_gui=False,
        num_seconds=num_second,
        begin_time=10,
        max_depart_delay=0,
        cav_env=enable_cv,  # has cav env
        collaborate=True,
    ))

    register_env("net", lambda _: my_env)

    env_config = {
        "get_additional_info": False,
    }

    config = None
    if run == "PPO":
        config = PPOConfig()
    elif run == "A3C":
        config = A3CConfig()
        # config = A3CConfig().training(model={"use_lstm": True}) 
    elif run == "SAC":
        config = SACConfig()
    elif run == "DQN":
        config = DQNConfig().training(n_step=5, noisy=True, num_atoms=5, v_min=-1000, v_max=10)
    assert config is not None

    config = (config.environment("net", env_config={"env_config": env_config})
              .framework("torch")
              .rollouts(rollout_fragment_length=128, num_rollout_workers=3)
              .training(train_batch_size=12000,model={'fcnet_hiddens': [256, 256]})
              .evaluation(
        evaluation_parallel_to_training=True,
        evaluation_num_workers=2,
        evaluation_interval=5,
        evaluation_duration="auto",
        evaluation_duration_unit="episodes",
    )
              .multi_agent(policies=my_env._agent_ids, policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id))
              .resources(num_gpus=1, num_cpus_per_worker=1))

    reporter = CLIReporter(
        metric_columns=["total_loss", "training_iteration", "episode_reward_mean", "timesteps_this_iter",
                        "timesteps_total", "episodes_total"]
    )

    print("params:", config.to_dict())

    results = tune.Tuner(
        run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            verbose=3,
            checkpoint_config=air.CheckpointConfig(num_to_keep=2, checkpoint_frequency=1,),
            progress_reporter=reporter,
            # storage_path=f'/data/folder1/folder2'  # change output folder by changing it
        ),
    ).fit()

    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    for n in args.net:
        for a in args.arch:
            for p in args.pr:
                main(a, n, p)