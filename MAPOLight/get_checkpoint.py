# get the best checkpoint in a trial
# example: python get_checkpoint.py <trial path>
# output: Checkpoint(local_path=<trial path>/checkpoint_003602)
# tell us that checkpoint_003602 has the highest `episode_reward_mean`

from ray.tune import Tuner
import sys

trial_path = sys.argv[1]
print(f"path: {trial_path}")

c = Tuner.restore(path=trial_path, trainable=trial_path.split('/')[-1]).get_results().get_best_result().get_best_checkpoint("episode_reward_mean", "max")
print(c)
