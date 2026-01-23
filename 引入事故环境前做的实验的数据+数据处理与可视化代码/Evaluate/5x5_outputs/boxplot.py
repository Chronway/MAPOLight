import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', 'a3c', 'sac', "fixed_time", "random", "maxpressure", "websters"], nargs='+')
# parser.add_argument('-g', '--graph', dest='graph', help='', default=['line', 'box'])
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.1", "0.05", "0.2", "fixed", "random", "maxpressure", "websters"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=True)
parser.add_argument('--range', dest='range', type=int, help='', default=0)


args = parser.parse_args()

def read_bunch(path, arch, num):
    data = pd.DataFrame(columns=['index', 'method', 'pr', args.type])
    for n in range(num):
        try:
            _data = pd.read_csv(f"{path}/{arch}_run{n}.csv")
            if args.range != 0:
                _data = _data[_data["step_time"] <= args.range]
            _data = _data[args.type].mean()
            d = pd.DataFrame({'index': n, "method": arch, 'pr': path, args.type: _data}, index=[0])
            data = pd.concat([data, d], ignore_index=True)
        except FileNotFoundError:
            print(f"在 {path} 未找到 {path}/{arch}_run{n}.csv，忽略 ")
    # print(data)
    return data


tbl = pd.DataFrame(columns=['index', 'method', 'pr', args.type])
for a in args.arch:
    for p in args.path:
        # print(f"{a}:{p}")
        tbl = pd.concat([tbl, read_bunch(p, a, args.num)], ignore_index=True)

if args.random:
    tbl = pd.concat([tbl, read_bunch("random", "random", args.num)], ignore_index=True)


tbl['method/pr'] = tbl['method'] + '/' + tbl['pr']
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'random/random' else 'random')

# print(tbl)
# tbl = tbl.pivot_table(index='index', columns='method/pr', values='wait_time_mean', aggfunc='first')
# sns.set_theme()
if args.type == 'total_stopped':
    tbl['total_stopped'] /= 25  # 平均到每个路口
fig = sns.boxplot(x="method/pr", y=args.type, hue="method/pr", data=tbl, dodge=False, linewidth=0.5, showfliers = False, width=0.5, legend=True)
fig.set(xlabel=None)
fig.tick_params(bottom=False)
fig.set(xticklabels=[])
# print(fig.containers)
# fig.bar_label(fig.containers[0])
# plt.ylabel("Average wait time (sec)")
plt.xlabel("Algorithm / PR")
if args.show:
    plt.show()
else:
    plt.savefig(f"boxplot.png", dpi=300)
