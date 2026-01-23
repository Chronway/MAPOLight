import os

os.environ["LIBSUMO_AS_TRACI"] = "True"

from ray.tune.registry import get_trainable_cls
from utils.mypettingzoo import MyPettingZooEnv
from ray.tune.registry import register_env
from env.SignalEnv import env
import numpy as np
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net")
parser.add_argument("--pr")
parser.add_argument("--collaborate", action="store_true")
# parser.add_argument("--accident-edge", nargs="+", default=["-gneE10"])
parser.add_argument("--accident-edge", nargs="+")
parser.add_argument("--accident-time", nargs="+")
parser.add_argument("--algorithm")
parser.add_argument("--extra-module")
parser.add_argument("--fcd", action="store_true")
parser.add_argument("--seed")
parser.add_argument("--scale", default="1")

parser.add_argument("--checkpoint", required=True)
parser.add_argument("--width128", action="store_true")
args = parser.parse_args()
print(args)

output_folder = "accident_outputs2"
assert args.accident_edge is None or len(args.accident_edge) == len(args.accident_time)

net_mapping = {
    "4x4": {
        "net_file": "sumo_files/4gridnet.net.xml",
        "route_file": "sumo_files/4groutes.xml",
        "num_seconds": 2400,
    },
    "5x5": {
        "net_file": "sumo_files/5_2net.net.xml",
        "route_file": "sumo_files/5_2groutes.xml",
        "num_seconds": 2400,
    },
    "2x2": {
        "net_file": "sumo_files/2gridnet.net.xml",
        "route_file": "sumo_files/2groutes.xml",
        "num_seconds": 3600,
    },
    "single": {
        "net_file": "sumo_files/single.net.xml",
        "route_file": "sumo_files/1groutes.xml",
        "num_seconds": 3600,
    }
}

my_env = MyPettingZooEnv(env(
    # net_file="sumo_files/5_2net.net.xml",
    # route_file="sumo_files/5_2groutes.xml",
    net_file=net_mapping[args.net]["net_file"],
    route_file=net_mapping[args.net]["route_file"],
    pr=float(args.pr),
    use_gui=False,
    num_seconds=net_mapping[args.net]["num_seconds"],
    begin_time=10,
    max_depart_delay=0,
    reward_fn="average-speed",
    cav_env=float(args.pr) != 1.,
    collaborate=args.collaborate,
    accident_edge=args.accident_edge,
    accident_time=args.accident_time,
    accident_duration=1200,
    fcd_path=f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}_fcd" if args.fcd else None,
    scale = args.scale
))

register_env("4x4grid", lambda _: my_env)

if args.algorithm == "ppo":
    config = get_trainable_cls(args.algorithm.upper()).get_default_config()

    config = (
        config.environment("4x4grid", env_config={"env_config": {"get_additional_info": True, }})
        .framework("torch")
        .rollouts(rollout_fragment_length=1024, num_rollout_workers=3)
        .training(train_batch_size=12000,
                  model={"fcnet_hiddens": [128, 128] if args.width128 else [256, 256],
                         "use_lstm": True if args.extra_module == "lstm" else False,
                         "use_attention": True if args.extra_module == "attention" else False})
        .evaluation(
            evaluation_parallel_to_training=True,
            evaluation_num_workers=2,
            evaluation_interval=5,
            evaluation_duration="auto",
            evaluation_duration_unit="episodes",
        )
        .multi_agent(policies=my_env._agent_ids,
                     policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id))
        .resources(num_gpus=0)
    )

    algo = config.build()

    algo.restore(args.checkpoint)

n = 1
def run_normal():
    import torch
    import random
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    for t in range(n):
        terminated = {"__all__": False}
        obs, _ = my_env.reset(seed=args.seed, options={"run": t})
        # state = {i: algo.get_policy(i).model.get_initial_state() for i in my_env._agent_ids}
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                action[agent_id] = algo.compute_single_action(
                    observation=obs[agent_id],
                    # state=state[agent_id],
                    explore=True,  # !!对于策略梯度类算法，反而要开启该标志，这样才能按照策略输出的分布进行采样
                    policy_id=agent_id,
                )
            obs, reward, terminated, truncate, info = my_env.step(action)

        # rews = 0
        # for i in my_env.sub_env.unwrapped.summary:
        #     rews += i["avg_speed"]
        # print(f"e-speed: {rews}")
        # rews = 0
        # for i in my_env.sub_env.unwrapped.summary:
        #     rews += i["total_speed"]
        # print(f"t-speed: {rews}")
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/")


def run_fix_time():
    np.random.seed(int(args.seed))
    for t in range(n):
        # 固定时间的动作顺序,因为每次决策间隔5秒，所以以下定义是每个相位持续30秒
        action_table = {
            i: itertools.cycle([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
            for i in my_env.get_agent_ids()
        }
        action_table = dict(sorted(action_table.items()))  # 排序键
        # 避免所有信号灯同时做同一个动作
        for c, i in enumerate(action_table):
            print(i)
            for _ in range(c * 6):
                next(action_table[i])
        obs, _ = my_env.reset(seed=args.seed, options={"run": t})
        terminated = {"__all__": False}
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                action[agent_id] = next(action_table[agent_id])
            obs, reward, terminated, truncate, info = my_env.step(action)

        
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/")


def run_random():
    for t in range(n):
        episode_reward = 0
        obs, _ = my_env.reset(seed=args.seed, options={"run": t})
        terminated = {"__all__": False}
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                action[agent_id] = my_env.action_space_sample([agent_id])[agent_id]
            obs, reward, terminated, truncate, info = my_env.step(action)
            for i in reward:
                episode_reward += reward[i]

        print(f"episode_reward: {episode_reward}")
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/")



def run_lstm():
    for t in range(n):
        episode_reward = 0
        terminated = {"__all__": False}
        obs, _ = my_env.reset(options={"run": t})
        state = {i: algo.get_policy(i).model.get_initial_state() for i in my_env._agent_ids}
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                action[agent_id], state[agent_id], _ = algo.compute_single_action(
                    observation=obs[agent_id],
                    state=state[agent_id],
                    explore=True,  # !!对于策略梯度类算法，反而要开启该标志，这样才能按照策略输出的分布进行采样
                    policy_id=agent_id,
                )
            obs, reward, terminated, truncate, info = my_env.step(action)
            for i in reward:
                episode_reward += reward[i]

        print(f"episode_reward: {episode_reward}")
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{args.accident_edge}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{args.accident_edge}_{args.algorithm}_{args.extra_module}/")

def run_attention():
    for t in range(n):
        episode_reward = 0
        terminated = {"__all__": False}
        obs, _ = my_env.reset(options={"run": t})
        num_transformers = config["model"]["attention_num_transformer_units"]
        memory_inference = config["model"]["attention_memory_inference"]
        attention_dim = config["model"]["attention_dim"]
        state = [
            np.zeros([memory_inference, attention_dim], np.float32)
            for _ in range(num_transformers)
        ]
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                action[agent_id], state[agent_id], _ = algo.compute_single_action(
                    observation=obs[agent_id],
                    state=state[agent_id],
                    explore=True,  # !!对于策略梯度类算法，反而要开启该标志，这样才能按照策略输出的分布进行采样
                    policy_id=agent_id,
                )
            obs, reward, terminated, truncate, info = my_env.step(action)
            for i in reward:
                episode_reward += reward[i]

        print(f"episode_reward: {episode_reward}")
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{args.accident_edge}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{args.accident_edge}_{args.algorithm}_{args.extra_module}/")

def run_maxp():
    np.random.seed(int(args.seed))
    for t in range(n):
        obs, _ = my_env.reset(seed=args.seed, options={"run": t})
        terminated = {"__all__": False}
        while not terminated["__all__"]:
            action = {}
            for agent_id in obs:
                # print(my_env.sub_env.max_pressure_action(agent_id))
                action[agent_id] = my_env.sub_env.max_pressure_action(agent_id)
            obs, reward, terminated, truncate, info = my_env.step(action)

        
        os.makedirs(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/",
            exist_ok=True)
        my_env.sub_env.unwrapped.save_csv(
            f"{output_folder}/{args.net}_{args.pr}_{args.collaborate}_{'_'.join(args.accident_edge) if args.accident_edge is not None else None}_{args.algorithm}_{args.extra_module}/")



if args.algorithm == "fix-time":
    run_fix_time()
elif args.algorithm == "random":
    run_random()
elif args.algorithm == "maxp":
    run_maxp()
else:
    if args.extra_module is None:
        run_normal()
    elif args.extra_module == "lstm":
        run_lstm()
    elif args.extra_module == "attention":
        run_attention()


my_env.close()
