import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(prog='box_and_line2', description='', epilog='')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo'], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["."], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--range', dest='range', type=int, help='', default=0)

parser.add_argument('--temp', dest='temp', action="store_true", default=False)

args = parser.parse_args()

if not args.temp:  # 不使用缓存
    def create_df(kind='time'):
        # 创建DataFrame并设置各列的类型
        if kind == 'time':  # 随时间的全部数据
            data = pd.DataFrame(columns=['arch', 'path', 'step_time', 'avg_wait_time', 'total_stopped', 'avg_speed'])
            data['arch'] = data['arch'].astype(str)
            data['path'] = data['path'].astype(str)
            data['step_time'] = data['step_time'].astype(float)
            data['avg_wait_time'] = data[args.type].astype(float)
            data['total_stopped'] = data[args.type].astype(float)
            data['avg_speed'] = data[args.type].astype(float)
    
    def read_bunch(path, arch, num):
        # 在指定路径下读取一组文件，如ppo_run0.csv, ppo_run1.csv, ..., ppo_run9.csv
        # 并将结果拼接成一个DataFrame
        time_data = create_df(kind='time')
        for n in range(num):
            try:
                tmp_data = pd.read_csv(f"{path}/{arch}_run{n}.csv")
                if args.range != 0:  # 裁剪数据到指定模拟时长
                    tmp_data = tmp_data[tmp_data["step_time"] <= args.range]
                tmp_data['arch'] = arch
                tmp_data['path'] = path
                time_data = pd.concat([time_data, tmp_data[['arch', 'path', 'step_time', 'avg_wait_time', 'total_stopped', 'avg_speed']]])
            except FileNotFoundError:
                pass
        return time_data, mean_data


    time_data = create_df(kind='time')
    mean_data = create_df(kind='mean')
    for a in args.arch:
        for p in args.path:
            t_data, m_data = read_bunch(p, a, args.num)
            time_data = pd.concat([time_data, t_data], ignore_index=True)
            mean_data = pd.concat([mean_data, m_data], ignore_index=True)

    print("load data finish.")
    # print(time_data)
    # print(mean_data)
    # exit()

    def post_process(data):
        # 数据后处理，给数据表添加便于画图的属性
        path2pr = {"1.0": "FO", "0.2": "20%", "0.1": "10%", "0.05": "5%"}
        arch2method = {  # 名称对应表
            "a3c": "MAA3C",
            "ppo": "MAPPO",
            "sac": "MASAC",
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

        data['total_stopped'] /= 25  # 将qlength指标平均到每个路口

    post_process(time_data)
    post_process(mean_data)
    time_data.to_csv('tmp_t.csv', index=False)
    mean_data.to_csv('tmp_m.csv', index=False)
else:  # 使用缓存
    time_data = pd.read_csv('tmp_t.csv', dtype={"arch": str, "path": str, "step_time": float, 'avg_wait_time': float, 'total_stopped': float, 'avg_speed': float, 'method': str, 'pr': str, 'category': str})
    mean_data = pd.read_csv('tmp_m.csv', dtype={"arch": str, "path": str, 'avg_wait_time': float, 'total_stopped': float, 'avg_speed': float, 'method': str, 'pr': str, 'category': str})

# exit()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 创建两个坐标系
# 设置显示位置
ax[0].set_position([0.12, 0.1, 0.2, 0.8])  # 箱图
ax[1].set_position([0.4, 0.1, 0.5, 0.8])  # 折线图

# 设置各类别颜色
palette_table = {
    "MAPPO/FO": "#00f",
    "MAPPO/20%": "#0050ff",
    "MAPPO/10%": "#0096ff",
    "MAPPO/5%": "#00d2ff",
    "MAA3C/FO": "#7f00ff",
    "MAA3C/20%": "#93f",
    "MAA3C/10%": "#b266ff",
    "MAA3C/5%": "#c9f",
    "MASAC/FO": "#ff00ff",
    "MASAC/20%": "#f3f",
    "MASAC/10%": "#f6f",
    "MASAC/5%": "#f9f",
    "Fixed time": "#ff7f00",
    "MaxPressure": "#cc0",
    "Websters": "#7fff00"
}

fig = sns.boxplot(y="category", x=args.type, hue='category', orient='y', palette=palette_table, data=mean_data, dodge=False, linewidth=0.5, showfliers = False, width=0.5, legend=None, ax=ax[0])
fig.set(xlabel=None)
fig.set(ylabel=None)
if args.type == 'avg_wait_time':
    fig.set_title('Average wait time (sec)')
elif args.type == 'avg_speed':
    fig.set_title('Average speed (m/s)')
elif args.type == 'total_stopped':
    fig.set_title('Average queue length (veh)')
elif args.type == 'reward':
    fig.set_title('Reward')



fig = sns.lineplot(data=time_data, x="step_time", y=args.type, hue='category', n_boot=32, ax=ax[1], palette=palette_table)
fig.get_legend().set_title("method(/pr)")
# plt.setp(ax[1].lines, alpha=.8)  # 设置折线图的透明度
# fig.set_yscale('log', base=2)
# fig.set_ylim(0.1, 120)

if args.type == 'avg_wait_time':
    fig.set(xlabel='Simulation time (sec)', ylabel='Average wait time (sec)')
elif args.type == 'avg_speed':
    fig.set(xlabel='Simulation time (sec)', ylabel='Average speed (m/s)')
elif args.type == 'total_stopped':
    fig.set(xlabel='Simulation time (sec)', ylabel='Average queue length (veh)')
elif args.type == 'reward':
    fig.set(xlabel='Simulation time (sec)', ylabel='Reward')


# plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig(f"boxplot.png", dpi=300)
