from math import hypot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', 'a3c', 'sac', "fixed_time", "random", "maxpressure", "websters"], nargs='+')
# parser.add_argument('-g', '--graph', dest='graph', help='', default=['line', 'box'])
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.1", "0.05", "0.2", "fixed", "random"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped', 'reward', 'corr'])
parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--range', dest='range', type=int, help='', default=0)


args = parser.parse_args()

print(args)

arch2name = {
    "a3c": "MA3C",
    "ppo": "MAPPO",
    "sac": "MASAC",
    "maxpressure": "MaxPressure",
    "sotl": "sotl",
    "websters": "websters",
}

n = args.num

if args.type != "corr":
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

                if path == "maxpressure" or path == "websters" or path == "fixed" or path == "random":
                    _data.insert(0, 'method', f"{arch2name.get(a, a)}")
                else:
                    _data.insert(0, 'method', f"{arch2name.get(a, a)}{f'_{path}' if len(args.path) > 1 else ''}")
                if data is None:
                    data = _data
                else:
                    data = pd.concat([data, _data])
            except FileNotFoundError:
                print(f"在 {path} 未找到 {a} 数据，忽略 ")

    if args.range != 0:
        data = data[data['step_time'] <= args.range]
    data = data.fillna(0)
    # exit()
    print("painting...")
    if args.type == 'total_stopped':
        data['total_stopped'] /= 8  # 平均到每个路口
    # print(data)
    data = data.groupby('method', as_index=False).mean()
    # print(data)
    # print(data['method'])
    # print(data[args.type])
    print(data)
    fig = sns.barplot(data=data, x="method", y=args.type, hue="method", legend=True)
    plt.xticks([])
    # fig.legend()
    if args.type == 'avg_wait_time':
        fig.set(ylabel='Average wait time (sec)')
    elif args.type == 'avg_speed':
        fig.set(ylabel='Average speed (m/s)')
    elif args.type == 'total_stopped':
        fig.set(ylabel='Average queue length (veh)')
    elif args.type == 'reward':
        fig.set(ylabel='Reward')
    
    # plt.legend()
elif args.type == "corr":
    finally_data = pd.DataFrame(columns=['cv', 'step_time', 'correlation_obs'])
    data = pd.DataFrame(columns=['step_time', 'correlation_obs'])
    for path in args.path:
        for a in args.arch:
        
            try:
                for i in range(0, n):
                    _data = pd.read_csv(f"{path}/{a}_run{i}.csv")[['step_time', 'correlation_obs']]
                    data = pd.concat([data, _data])
            except FileNotFoundError:
                print(f"在 {path} 未找到 {a} 数据，忽略 ")
            except KeyError:
                print(f"在 {path}/{a} 中未找到 correlation_obs 数据，忽略")
                
        data.insert(0, 'penalty rate', path)
        finally_data = pd.concat([finally_data, data])
        data = pd.DataFrame(columns=['step_time', 'correlation_obs'])
        
    print(finally_data)
    exit()
    print("painting...")
    fig = sns.barplot(data=finally_data, x="method/pr", y="correlation_obs", hue="penalty rate")
    # plt.legend()
# plt.show()
    plt.xlabel("Simulation time (s)")
    plt.ylabel("correlation")
# plt.xlim(0, args.range)

if args.show:
    plt.show()
else:
    plt.savefig(f"{args.arch}-{args.path}-{args.type}.png", dpi=300)
