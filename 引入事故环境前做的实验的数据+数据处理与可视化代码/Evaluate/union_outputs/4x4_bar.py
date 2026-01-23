import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(prog='', description='', epilog='')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('--net', dest='net', help='', default=["4x4"], nargs='+')
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo',"a3c", "sac", "uniform", "maxpressure", "websters", "random"], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.2", "0.1", "0.05", "uniform", "maxpressure", "websters", "random"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default=['avg_wait_time', 'avg_speed', 'total_stopped'], nargs="+")
parser.add_argument('-s', '--save', dest='save', default=None)
parser.add_argument('-w', '--width', dest='width', type=float, default=.8)
parser.add_argument('--range', dest='range', type=int, help='', default=0)

args = parser.parse_args()

def create_df():
    data = pd.DataFrame(columns=['net', 'arch', 'path', 'avg_wait_time', 'total_stopped', 'avg_speed'])
    data['net'] = data['net'].astype(str)
    data['arch'] = data['arch'].astype(str)
    data['path'] = data['path'].astype(str)
    data['avg_wait_time'] = data['avg_wait_time'].astype(float)
    data['total_stopped'] = data['total_stopped'].astype(float)
    data['avg_speed'] = data['avg_speed'].astype(float)
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
    for a in args.arch:
        for p in args.path:
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

type2name = {
    "avg_wait_time": "Average wait time",
    "avg_speed": "Average speed",
    "total_stopped": "Average queue length",
}
def post_post_process(data):
    origin = data.copy()
    data['metric_catalog'] = type2name[args.type[0]]
    data['metric'] = data[args.type[0]]
    for i in args.type[1:]:
        copyed = origin.copy()
        copyed['metric_catalog'] = type2name[i]
        copyed['metric'] = origin[i]
        data = pd.concat([data, copyed], ignore_index=True)
    return data

mean_data = post_post_process(mean_data)
# print(time_data)
print(mean_data)
# exit()
data = mean_data


# 设置各类别颜色
palette_table = {
    "MAPO-PPO/FO": "#00f",
    "MAPO-PPO/100%": "#00f",
    "MAPO-PPO/20%": "#0050ff",
    "MAPO-PPO/10%": "#0096ff",
    "MAPO-PPO/5%": "#00d2ff",
    "MAPO-A3C/FO": "#7f00ff",
    "MAPO-A3C/100%": "#7f00ff",
    "MAPO-A3C/20%": "#93f",
    "MAPO-A3C/10%": "#b266ff",
    "MAPO-A3C/5%": "#c9f",
    "MAPO-SAC/FO": "#ff00ff",
    "MAPO-SAC/100%": "#ff00ff",
    "MAPO-SAC/20%": "#f3f",
    "MAPO-SAC/10%": "#f6f",
    "MAPO-SAC/5%": "#f9f",
    "Fixed time": "#ff0000",
    "MaxPressure": "#70db93",
    "Websters": "#ffa500",
    "Random": "#999999"
}


plt.rcParams["figure.figsize"] = (18, 10)

ax = sns.barplot(data=data, x='metric_catalog', y='metric', hue='category', palette=palette_table,
                err_kws={"linewidth": .9}, capsize=.3, width=args.width,
                edgecolor='.2',
                legend="full")
if True:
    ax.set_yscale('log', base=10)
    # ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:g}'.format(x)))  # 将坐标轴label设定为十进制
ax.set(xlabel=None)
ax.set(ylabel=None)

# 设置柱子上显示的标签
# for i in ax.containers:
#     ax.bar_label(i, fmt="%.1f", fontsize=5, label_type='center')
ax.get_legend().set_title("")
ax.set_xticks(list(range(len(ax.get_xticklabels()))))  # 用于避免warning，可以去掉
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")  # 设置x坐标标签的对齐和旋转
ax.set_position([0.05, 0.05, 0.6, 0.6])  # 设置图像位置，以便x标签显示完整
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.01))  # 将legend移动到图外，以免遮挡图像

if args.type == 'avg_wait_time':
    ax.set_title('Average wait time (sec)')
elif args.type == 'avg_speed':
    ax.set_title('Average speed (m/s)')
elif args.type == 'total_stopped':
    ax.set_title('Average queue length (veh)')


if args.save:
    plt.savefig(f"{args.save}", dpi=300, bbox_inches='tight')
else:
    plt.show()
