# 该example基于ray 2.6.3，迁移到新版可能要做适配
import os

os.environ["LIBSUMO_AS_TRACI"] = "True"  # 加速

from ray.rllib.algorithms.ppo import PPOConfig

from ray.tune.registry import register_env
from ray.tune import Tuner

from ray.air import RunConfig, CheckpointConfig

from env.SignalEnv import env as signal_env

from utils.mypettingzoo import MyPettingZooEnv
import os

# net_file = "sumo_files/2gridnet.net.xml"
net_file = os.path.abspath("sumo_files/single.net.xml")
route_file = os.path.abspath("sumo_files/1groutes.xml")
co = False

dummy_env = signal_env(
    net_file=net_file,
    route_file=route_file,
    pr=1,
    use_gui=False,  # 是否显示gui
    begin_time=10,  # 智能体开始介入的时间
    num_seconds=3600,  # 模拟的总时间
    max_depart_delay=0,  # sumo参数，见sumo文档
    time_to_teleport=-1,  # sumo参数，见sumo文档
    delta_time=5,  # 智能体决策的间隔时间
    yellow_time=3,  # 黄灯的持续时间
    min_green=5,  # 最小绿灯持续时间，如果当前动作选择改变相位，但当前的绿灯未持续足够时间，则当前动作失效
    max_green=50,  # sumo-rl遗留代码，根本没用到
    reward_fn="average-speed",  # 奖励函数
    sumo_warnings=True,  # 是否显示sumo警告
    cav_env=False,  # 是否启动cav
    cav_compare=False,  # 是否启动cav比较，训练时用不到，且计算量大，训练时应恒为False
    collaborate=co,  # 是否启动智能体协作：使用协作观察和协作奖励
)
agents = dummy_env.possible_agents  # dummy_env纯粹是用来获取智能体列表的

register_env(
    "signal_env",
    lambda _: MyPettingZooEnv(
        signal_env(
            net_file=net_file,  # ray2.6.3在创建环境时必须使用绝对路径
            route_file=route_file,  # 因为其脑残代码会在评估时将相对路径判断成url
            pr=1,
            use_gui=False,  # 是否显示gui
            begin_time=10,  # 智能体开始介入的时间
            num_seconds=3600,  # 模拟的总时间
            max_depart_delay=0,  # sumo参数，见sumo文档
            time_to_teleport=-1,  # sumo参数，见sumo文档
            delta_time=5,  # 智能体决策的间隔时间
            yellow_time=3,  # 黄灯的持续时间
            min_green=5,  # 最小绿灯持续时间，如果当前动作选择改变相位，但当前的绿灯未持续足够时间，则当前动作失效
            max_green=50,  # sumo-rl遗留代码，根本没用到
            reward_fn="average-speed",  # 奖励函数
            sumo_warnings=True,  # 是否显示sumo警告
            cav_env=False,  # 是否启动cav
            cav_compare=False,  # 是否启动cav比较，训练时用不到，且计算量大，训练时应恒为False
            collaborate=co,  # 是否启动智能体协作：使用协作观察和协作奖励
            accident_num=2,
            accident_duration=600,
        )
    ),
)

register_env(
    "eval_signal_env",
    lambda _: MyPettingZooEnv(
        signal_env(
            net_file=net_file,  # ray2.6.3在创建环境时必须使用绝对路径
            route_file=route_file,  # 因为其内部代码会在评估时将相对路径判断成url
            pr=1,
            use_gui=False,  # 是否显示gui
            begin_time=10,  # 智能体开始介入的时间
            num_seconds=3600,  # 模拟的总时间
            max_depart_delay=0,  # sumo参数，见sumo文档
            time_to_teleport=-1,  # sumo参数，见sumo文档
            delta_time=5,  # 智能体决策的间隔时间
            yellow_time=3,  # 黄灯的持续时间
            min_green=5,  # 最小绿灯持续时间，如果当前动作选择改变相位，但当前的绿灯未持续足够时间，则当前动作失效
            max_green=50,  # sumo-rl遗留代码，根本没用到
            reward_fn="average-speed",  # 奖励函数
            sumo_warnings=True,  # 是否显示sumo警告
            cav_env=False,  # 是否启动cav
            cav_compare=False,  # 是否启动cav比较，训练时用不到，且计算量大，训练时应恒为False
            collaborate=co,  # 是否启动智能体协作：使用协作观察和协作奖励
            accident_edge=["-gneE10"],
            accident_duration=600,
            accident_time=["1800"],
        )
    ),
)

# 这是一个简易配置
# 这里的配置rllib每次更新都会破坏式地改一堆，要注意适配
config = (
    PPOConfig()
    .environment("signal_env")
    .training()
    .resources(num_gpus=1, num_gpus_per_learner_worker=1)
    .rollouts(num_rollout_workers=5)
    .multi_agent(
        policies=agents,
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id),
    )
    .evaluation(
        evaluation_interval=10,
        evaluation_duration=1,
        evaluation_config={"env": "eval_signal_env"},
    )  # 每2次迭代后跑1整个episode进行评估，实际训练中不需要这么频繁地进行评估
    # 相较于训练reward，评估reward或许更有意义
)

Tuner(
    trainable="PPO",
    param_space=config.to_dict(),
    run_config=RunConfig(
        verbose=3,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_frequency=10,  # 每一次迭代都保存checkpoint，实际训练中不需要这么频繁地保存checkpoint
        ),
        # storage_path=f'',
    ),
).fit()
