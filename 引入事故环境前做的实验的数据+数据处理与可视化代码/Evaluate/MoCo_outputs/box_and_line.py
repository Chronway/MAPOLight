import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', 'a3c', 'sac', "uniform", "random", "maxpressure", "websters"], nargs='+')
# parser.add_argument('-g', '--graph', dest='graph', help='', default=['line', 'box'])
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.1", "0.05", "uniform", "websters", "maxpressure"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--range', dest='range', type=int, help='', default=0)


args = parser.parse_args()
          
arch2name = {
    "a3c": "MA3C",
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

if args.random:
    tbl = pd.concat([tbl, read_bunch("random", "random", args.num)], ignore_index=True)


tbl['pr'][tbl['pr'] == '1.0'] = 'FO'
tbl['method/pr'] = tbl['method'].map(lambda a: arch2name.get(a, a)) + '/' + tbl['pr']
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'Random/random' else 'Random')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'uniform/uniform' else 'Fixed time')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'Websters/websters' else 'Websters')
tbl['method/pr'] = tbl['method/pr'].map(lambda a: a if a != 'MaxPressure/maxpressure' else 'MaxPressure')

# tbl.loc[tbl['method'] == "uniform", ['total_stopped']] /= 3.17
# tbl.loc[tbl['method'] == "websters", ['total_stopped']] /= 3.17
# tbl.loc[tbl['method'] == "maxpressure", ['total_stopped']] /= 3.17
# print(tbl)
# tbl = tbl.pivot_table(index='index', columns='method/pr', values='wait_time_mean', aggfunc='first')
# sns.set_theme()
if args.type == 'total_stopped':
    tbl['total_stopped'] /= 8  # 平均到每个路口
print(tbl)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].set_position([0.12, 0.1, 0.2, 0.8])
ax[1].set_position([0.4, 0.1, 0.5, 0.8])
fig = sns.boxplot(y="method/pr", x=args.type, orient='y', color='#AAAAAA', data=tbl, dodge=False, linewidth=0.5, showfliers = False, width=0.5, legend=None, ax=ax[0])
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



# lineplot


n = args.num

data = None
if args.random:
    try:
        _data = pd.read_csv(f"random/random_run0.csv")[['step_time', args.type]]
        # _data = _data[_data['step_time'] <= args.range]

        for i in range(1, n):
            _data = pd.concat([_data, pd.read_csv(f"random/random_run{i}.csv")[['step_time', args.type]]])
            # _data = _data[_data['step_time'] <= args.range]

        _data.insert(0, 'method', f"random")

        if data is None:
            data = _data
        else:
            data = pd.concat([data, _data])
    except FileNotFoundError:
        print(f"在 random 未找到 random 数据，忽略 ")

for a in args.arch:
    for path in args.path:
        try:
            _data = pd.read_csv(f"{path}/{a}_run0.csv")[['step_time', args.type]]
            # _data = _data[_data['step_time'] <= args.range]

            for i in range(1, n):
                try:
                    _data = pd.concat([_data, pd.read_csv(f"{path}/{a}_run{i}.csv")[['step_time', args.type]]])
                    # _data = _data[_data['step_time'] <= args.range]
                except FileNotFoundError:
                    print(f"在 {path} 未找到 {f'{a}_run{i}.csv'}，忽略 ")

            _data.insert(0, 'method', f"{arch2name.get(a, a)}{('/' + path if path != '1.0' else '/FO') if len(args.path) > 1 and path != 'random' and path != 'fixed' and path != 'maxpressure' and path != 'websters' else ''}")

            if data is None:
                data = _data
            else:
                data = pd.concat([data, _data])
        except FileNotFoundError:
            print(f"在 {path} 未找到 {a} 数据，忽略 ")

if args.range != 0:
    data = data[data['step_time'] <= args.range]
data = data.fillna(0)

data['method'] = data['method'].map(lambda a: a if a != 'uniform/uniform' else 'Fixed time')
print("painting...")
if args.type == 'total_stopped':
    data['total_stopped'] /= 8  # 平均到每个路口

print(tbl[tbl['method'] == 'a3c'])
print(data[data['method'] == 'a3c'])

fig = sns.lineplot(data=data, x="step_time", y=args.type, hue="method", n_boot=32, ax=ax[1])
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
