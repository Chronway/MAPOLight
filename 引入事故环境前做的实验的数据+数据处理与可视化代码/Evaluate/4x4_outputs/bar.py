import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', "websters", "maxpressure", "uniform"], nargs='+')
# parser.add_argument('-g', '--graph', dest='graph', help='', default=['line', 'box'])
parser.add_argument('-p', '--path', dest='path', help='', default=['1.0', '0.2', '0.1', '0.05', 'uniform', 'maxpressure', 'websters'], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--range', dest='range', type=int, help='', default=0)


args = parser.parse_args()
          
arch2name = {
    "a3c": "MAA3C",
    "ppo": "MAPPO",
    "sac": "MASAC",
    "maxpressure": "MaxPressure",
    "sotl": "sotl",
    "websters": "Websters",
    "fixed_time": "Fixed time",
    "random": "Random"
}


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


tbl['pr'][tbl['pr'] == '1.0'] = 'FO'
tbl['method/pr'] = tbl['method'].map(lambda a: arch2name.get(a, a)) + '/' + tbl['pr']
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'uniform/MoCo/uniform' else 'Fixed time/MoCo')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'uniform/4x4/uniform' else 'Fixed time/4x4')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'uniform/5x5/uniform' else 'Fixed time/5x5')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'MaxPressure/MoCo/maxpressure' else 'Maxpressure/MoCo')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'MaxPressure/4x4/maxpressure' else 'Maxpressure/4x4')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'MaxPressure/5x5/maxpressure' else 'Maxpressure/5x5')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'Websters/MoCo/websters' else 'Websters/MoCo')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'Websters/4x4/websters' else 'Websters/4x4')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'Websters/5x5/websters' else 'Websters/5x5')


if args.type == 'total_stopped':
    # print(tbl.apply(lambda r: div_num(r),axis=1))
    tbl['total_stopped'] = tbl['total_stopped'] / 16

    # exit(0)
tbl['environment'] = tbl['pr'].map(lambda a: a.split('/')[0])
print(tbl)
# exit()

fig = sns.barplot(x="method/pr", y=args.type, capsize=.2, hue="environment", errwidth=.5, orient='x', data=tbl, dodge='auto', linewidth=0.5, width=0.5, legend='full')

fig.set(xlabel=None)
fig.set(ylabel=None)
# fig.set_xticks([])
fig.set_xticklabels(fig.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
fig.set_position([0.2, 0.3, 0.6, 0.6])

if args.type == 'avg_wait_time':
    fig.set_title('Average wait time (sec)')
elif args.type == 'avg_speed':
    fig.set_title('Average speed (m/s)')
elif args.type == 'total_stopped':
    fig.set_title('Average queue length (veh)')
elif args.type == 'reward':
    fig.set_title('Reward')


# plt.tight_layout()
if args.show:
    plt.show()
else:
    plt.savefig(f"boxplot.png", dpi=300)
