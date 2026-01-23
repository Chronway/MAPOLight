# example:
# python test2.py -a random -o random 
# python test2.py -a random -o 5x5_random -n /sumo_files/5_2net.net.xml -r /sumo_files/5_2groutes.xml --cfg /sumo_files/5_2grid.sumocfg -t 4800
# python test2.py -a ppo -c /data/lyq/cav_0.05/PPO/PPO_4x4grid_6a03f_00000_0_2023-09-11_19-14-45/checkpoint_000520/ -p 0.05 --co 

import os
os.environ['LIBSUMO_AS_TRACI'] = "True"

from ray.tune.registry import register_env
from utils.mypettingzoo import MyPettingZooEnv
from env.SignalEnv import env
from ray.rllib.algorithms.algorithm import Algorithm
import argparse
import ray

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("-a", "--arch", dest="arch", help="method to test", default="random", choices=['ppo', 'a3c', 'sac', 'dqn', 'random'])
parser.add_argument("-c", "--checkpoint", dest="checkpoint", help="path to checkpoint. e.g.: ~/ray_results/PPO/PPO_net_34ccb_00000_0_2026-01-23_18-06-38/checkpoint_000036")
parser.add_argument("-o", "--output", dest="output", help="path to output folder", default=".")
parser.add_argument("-p", "--pr", dest="pr", type=float, help="penetration rates", default=1.0)
parser.add_argument("--co", "--collaborate", dest="collaborate", action="store_true", default=True, help="enable agents collaborate mode")
parser.add_argument("--128width", dest="_128width", action="store_true", default=False, help="legacy. For running original smaller model")
parser.add_argument("-n", "--net", dest="net", default="/sumo_files/moco.net.xml", help="network file")
parser.add_argument("-r", "--route", dest="route", default="/sumo_files/moco_jtr_out.rou.xml", help="route file")
parser.add_argument("--cfg", dest="cfg", default="/sumo_files/testmap.sumocfg", help="sumocfg file")
parser.add_argument("-t", "--time", dest="time", default=7300, type=int, help="time to simulated")

args = parser.parse_args()

print(args)

ray.init(num_cpus=3, num_gpus=1)

if args.arch.lower() == "ppo":
    from ray.rllib.algorithms.ppo import PPOConfig
elif args.arch.lower() == "a3c":
    from ray.rllib.algorithms.a3c import A3CConfig
elif args.arch.lower() == "sac":
    from ray.rllib.algorithms.sac import SACConfig
elif args.arch.lower() == "dqn":
    from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig

current_dir = "."

net_file = current_dir + args.net
rou_file = current_dir + args.route
cfg_file = current_dir + args.cfg

env = MyPettingZooEnv(
    env(
        net_file=net_file,
        route_file=rou_file,
        config_file=cfg_file,
        PR=args.pr,
        out_csv_name=None,
        # out_csv_name=current_dir + '/outputs/output_ppo',
        use_gui=False,
        num_seconds=args.time,
        # time_to_load_vehicles=10,
        begin_time=10,
        max_depart_delay=0,
        # ====cav====
        cav_env=True if args.pr != 1 else False,
        cav_compare=True if args.pr != 1 else False,
        # ===========
        collaborate=args.collaborate
    )
)

register_env("_", lambda _: env)


if args.arch.lower() != "random":
    # algo = Algorithm.from_checkpoint(args.checkpoint)
    if args.arch.lower() == "ppo":
        algo = PPOConfig().environment("_", env_config={"env_config": {"get_additional_info": True, }}).rollouts(
            num_rollout_workers=1).multi_agent(
            policies=env._agent_ids, policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: agent_id)).resources(num_gpus=1)
    elif args.arch.lower() == "a3c":
        algo = A3CConfig().environment("_", env_config={"env_config": {"get_additional_info": True, }}).rollouts(
            num_rollout_workers=1).multi_agent(
            policies=env._agent_ids, policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id)).resources(
            num_gpus=0)
    elif args.arch.lower() == "sac":
        algo = SACConfig().environment("_", env_config={"env_config": {"get_additional_info": True, }}).rollouts(
            num_rollout_workers=1).multi_agent(
            policies=env._agent_ids, policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id)).resources(
            num_gpus=0)
    elif args.arch.lower() == "dqn":
        algo = ApexDQNConfig()
        replay_config = algo.replay_buffer_config
        replay_config.update(
            {   
                "no_local_replay_buffer": False,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 10000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.5,
                "prioritized_replay_eps": 3e-6,
            }
        )
        algo = ApexDQNConfig().training(replay_buffer_config=replay_config, n_step=5, noisy=True, num_atoms=5, v_min=-1000, v_max=10)
        algo = algo.environment("_", env_config={"env_config": {"get_additional_info": True, }}).rollouts(
            num_rollout_workers=1).multi_agent(
            policies=env._agent_ids, policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id)).resources(
            num_gpus=0)
        
    if args._128width:
        algo.training(model={'fcnet_hiddens': [128, 128]})
    algo = algo.build()
    
    algo.restore(args.checkpoint)


n = 10
for t in range(n):
    episode_reward = 0
    done = {"__all__": False}
    obs, _ = env.reset()

    while not done["__all__"]:
        action = {}

        for agent_id in obs:
            if args.arch == "random":
                action = env.action_space_sample([agent_id])
            else:
                action[agent_id] = algo.compute_single_action(
                    observation=obs[agent_id],
                    explore=False,
                    policy_id=agent_id,
                )

        obs, reward, done, truncate, info = env.step(action)
        for i in reward:
            episode_reward += reward[i]


    print(f"episode_reward: {episode_reward}")
    os.makedirs(f"outputs/{args.output}/{args.pr}/", exist_ok=True)
    env.env.unwrapped.save_csv(f"outputs/{args.output}/{args.pr}/{args.arch}", t)
env.close()
