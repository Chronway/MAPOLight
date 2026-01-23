# 获取一次 trial 中指标最好的 checkpoint
# 用于训练完成后，找到效果最好的模型
# example: python get_checkpoint.py /data/lyq/yoro3/5x5/PPO/1/PPO
# 其中 /data/lyq/yoro3/5x5/PPO/1/PPO 是某次 trial 的保存路径
# 输出 Checkpoint(local_path=/data/lyq/yoro3/5x5/PPO/1/PPO/PPO_net_5628c_00000_0_2024-01-20_12-42-37/checkpoint_003602)
# 告诉我们第 3602 次迭代的模型是 episode_reward_mean 最高的

from ray.tune import Tuner
import sys

trial_path = sys.argv[1]
print(f"path: {trial_path}")

c = Tuner.restore(path=trial_path, trainable=trial_path.split('/')[-1]).get_results().get_best_result().get_best_checkpoint("episode_reward_mean", "max")
print(c)
