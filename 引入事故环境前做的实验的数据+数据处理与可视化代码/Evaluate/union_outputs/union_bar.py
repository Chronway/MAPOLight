import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='box_and_line2', description='优雅的绘图代码', epilog='')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('--net', dest='net', help='', default=["4x4", "5x5", "MoCo"], nargs='+')
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo',"uniform", "maxpressure", "websters"], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.2", "0.1", "0.05", "uniform", "maxpressure", "websters",], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--save', dest='save', default=None)
parser.add_argument('--range', dest='range', type=int, help='', default=0)
parser.add_argument('--temp', dest='temp', action="store_true", default=False)

args = parser.parse_args()

if not args.temp:  # 不使用缓存
    def create_df():
        data = pd.DataFrame(columns=['net', 'arch', 'path', 'avg_wait_time', 'total_stopped', 'avg_speed'])
        data['net'] = data['net'].astype(str)
        data['arch'] = data['arch'].astype(str)
        data['path'] = data['path'].astype(str)
        data['avg_wait_time'] = data[args.type].astype(float)
        data['total_stopped'] = data[args.type].astype(float)
        data['avg_speed'] = data[args.type].astype(float)
        return data
    
    def read_bunch(net, path, arch, num):
        # 在指定路径下读取一组文件，如ppo_run0.csv, ppo_run1.csv, ..., ppo_run9.csv
        # 并将结果拼接成一个DataFrame
        mean_data = create_df()
        for n in range(num):
            try:
                tmp_data = pd.read_csv(f"{net}/{path}/{arch}_run{n}.csv")
                if args.range != 0:  # 裁剪数据到指定模拟时长
                    tmp_data = tmp_data[tmp_data["step_time"] <= args.range]
                tmp_data['net'] = net
                tmp_data['arch'] = arch
                tmp_data['path'] = path
                mean_data.loc[len(mean_data)] = {"net": net, "arch": arch, "path": path, "avg_wait_time": tmp_data["avg_wait_time"].mean(), "total_stopped": tmp_data["total_stopped"].mean(), "avg_speed": tmp_data["avg_speed"].mean()}
            except FileNotFoundError:
                pass
        return mean_data


    mean_data = create_df()
    for n in args.net:
        for p in args.path:
            for a in args.arch:
                m_data = read_bunch(n, p, a, args.num)
                mean_data = pd.concat([mean_data, m_data], ignore_index=True)

    print("load data finish.")

    def post_process(data):
        # 数据后处理，给数据表添加便于画图的属性
        path2pr = {"1.0": "100%", "0.2": "20%", "0.1": "10%", "0.05": "5%"}
        arch2method = {  # 名称对应表
            "a3c": "MAPO-A3C",
            "ppo": "MAPO-PPO",
            "sac": "MAPO-SAC",
            "maxpressure": "MaxPressure",
            "websters": "Websters",
            "uniform": "Fixed time",
            "random": "Random",
        }

        data['method'] = data['arch'].apply(lambda x: arch2method.get(x, x))  # 新建method，用于存放算法的正式(显示)名称
        data['pr'] = data['path'].apply(lambda x: path2pr.get(x, None))  # 新建pr列，传统算法的pr是None
        data.loc[data['pr'].isnull(), 'category'] = data[data['pr'].isnull()]['method']  # 算法类别，应为 method/pr (强化学习) 或 method (传统方法)
        data.loc[data['pr'].notnull(), 'category'] = data[data['pr'].notnull()]['method'] + '/' +  data[data['pr'].notnull()]['pr']
        data.loc[data['pr'].isnull(), 'pr'] = "FO"  # 给传统算法补上pr字段

        # 将qlength指标平均到每个路口
        data.loc[data['net'] == '4x4', 'total_stopped'] /= 16  
        data.loc[data['net'] == '5x5', 'total_stopped'] /= 25
        data.loc[data['net'] == 'MoCo', 'total_stopped'] /= 8

    post_process(mean_data)
    mean_data.to_csv('tmp_m.csv', index=False)
else:  # 使用缓存
    mean_data = pd.read_csv('tmp_m.csv', dtype={"arch": str, "path": str, 'avg_wait_time': float, 'total_stopped': float, 'avg_speed': float, 'method': str, 'pr': str, 'category': str})

# print(time_data)
print(mean_data)
data = mean_data


# 设置各类别颜色
palette_table = {
    "MAPO-PPO/FO": "#00f",
    "MAPO-PPO/100%": "#00f",
    "MAPO-PPO/20%": "#0050ff",
    "MAPO-PPO/10%": "#0096ff",
    "MAPO-PPO/5%": "#00d2ff", 
    "Fixed time": "#ff7f00",
    "MaxPressure": "#ffff2c",
    "Websters": "#aeff00"
}

ax = sns.barplot(data=data, x='net', y=args.type, hue='category', palette=palette_table,
                err_kws={"linewidth": .9}, capsize=.3,
                edgecolor='.2',
                legend="full")
if args.type == 'avg_wait_time':
    ax.set_yscale('log', base=10)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:g}'.format(x)))  # 将坐标轴label设定为十进制
ax.set(xlabel=None)
ax.set(ylabel=None)

# 设置柱子上显示的标签
for i in ax.containers:
    ax.bar_label(i, fmt="%.1f", fontsize=5, label_type='center')
ax.get_legend().set_title("")
ax.set_xticks(list(range(len(ax.get_xticklabels()))))  # 用于避免warning，可以去掉
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")  # 设置x坐标标签的对齐和旋转
ax.set_position([0.1, 0.3, 0.6, 0.6])  # 设置图像位置，以便x标签显示完整
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.025))  # 将legend移动到图外，以免遮挡图像

if args.type == 'avg_wait_time':
    ax.set_ylabel('Average wait time (sec)')
elif args.type == 'avg_speed':
    ax.set_ylabel('Average speed (m/s)')
elif args.type == 'total_stopped':
    ax.set_ylabel('Average queue length (veh)')


if args.save is None:
    plt.show()
else:
    plt.savefig(f"{args.save}", dpi=300, bbox_inches='tight')
