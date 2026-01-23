import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo'], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["0.2", "0.1", "0.05"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="corr", choices=['corr'])
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--save', default="Corr.pdf")
parser.add_argument('--range', dest='range', type=int, help='', default=0)


args = parser.parse_args()

print(args)

n = args.num
finally_data = pd.DataFrame(columns=['cv', 'step_time', 'correlation_obs'])
data = pd.DataFrame(columns=['step_time', 'correlation_obs'])

path2pr = {
    '0.2': "20%",
    '0.1': "10%",
    '0.05': "5%",
}

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
            
    data.insert(0, 'CV P-Rate', path2pr[path])
    finally_data = pd.concat([finally_data, data])
    data = pd.DataFrame(columns=['step_time', 'correlation_obs'])
    

print("painting...")
print(finally_data)
fig = sns.lineplot(data=finally_data, x="step_time", y="correlation_obs", hue="CV P-Rate", n_boot=32)
fig.axhline(y=0.9, color='red', linestyle='--')
# plt.legend()
# plt.show()
plt.xlabel("Simulation time (s)")
plt.ylabel("correlation")
if args.range != 0:
    plt.xlim(0, args.range)
if args.show:
    plt.show()
else:
    plt.savefig(f"{args.save}", dpi=300)
