from utils import *
import seaborn as sns
import glob
import matplotlib.pyplot as plt

label_table = {
    "4x4_1_True_B2C2_D3C3_D3D2_fix-time_None": "Fixed-time TLSC with accident",
    "4x4_1_True_B2C2_D3C3_D3D2_ppo_None": "MARL TLSC with accident",
    "4x4_1_True_None_fix-time_None": "Fixed-time TLSC without accident",
    "4x4_1_True_None_ppo_None": "MARL TLSC without accident",
}
def draw_once(path):
    batch = read_batch(path)
    sns.lineplot(batch, x="step_time", y="avg_speed", label=label_table[path])
    


for i in glob.glob("4x4*_1_*None"):
    draw_once(i)

plt.xlim(0, 2000)
plt.ylim(0, 15)
# plt.show()
plt.xlabel("Simulation Time")
plt.ylabel("Average speed(m/s)")
plt.savefig("比较PPO与固定时间在有无事故下的速度.svg", dpi=300)
