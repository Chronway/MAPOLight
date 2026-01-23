# python .\lineplot.py -f A3C_1 A3C_0.05 A3C_0.1 -c nocv cv0.05 cv0.1 -x 30000000 -l pr --title "A3C train reward" --ltitle cv -o A3C --show
# python .\lineplot.py -f PPO_1 PPO_0.05 PPO_0.1 PPO_0.2 -c nocv cv0.05 cv0.1 cv0.2 -x 30000000 -l pr --title "PPO train reward" --ltitle cv -o PPO --show
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does', epilog='Text at the bottom of help')

# parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-f', '--files', dest='files', help='', nargs='+')
parser.add_argument('-c', '--cv', dest='hue', help='', nargs='+')
parser.add_argument('-o', '--output', dest='output', help='')
parser.add_argument('-l', '--legend', dest='legend', help='')
parser.add_argument('--title', dest='title', help='')
parser.add_argument('--ltitle', dest='ltitle', help='')
parser.add_argument('--log', action="store_true", dest='log', help='', default=False)
parser.add_argument('--show', action="store_true", dest='show', help='', default=False)
parser.add_argument('-x', '--xlimit', type=float, dest='xlimit', help='')
parser.add_argument('-y', '--ylimit', type=float, dest='ylimit', help='')

# parser.add_argument('-p', '--path', dest='path', help='', default=["."], nargs='+')
# parser.add_argument('-t', '--type', dest='type', help='', default="wait_time", choices=['wait_time', 'corr'])
# parser.add_argument('-r', '--random', dest='random', action="store_true", help='', default=False)
# parser.add_argument('--range', dest='range', type=int, help='', default=1600)

from collections import Counter
def find_common_max_element(arr):
    count_dict = Counter(arr)
    max_element, max_count = count_dict.most_common(1)[0]
    return max_element

args = parser.parse_args()

data = pd.DataFrame(columns=['Step', 'Value'])
for f_name, f_hue in zip(args.files, args.hue):
    _data = pd.read_csv(f_name + ".csv")[['Step', 'Value']]
    _data.insert(0, args.legend, f_hue)
    data = pd.concat([_data, data])

data = data[data['Step'] <= args.xlimit]
# for i in set(args.hue):
#     _data = data[data[args.legend] == i]
#     common_max = find_common_max_element(list(_data['Step']))
#     _data = _data[_data['Step'] <= common_max]
print(data)
sns.lineplot(data=data, x='Step', y='Value', hue=args.legend)
plt.xlim(0, args.xlimit)
plt.ylim( args.ylimit, 0)
# plt.title(f"{args.output} train reward")

if args.log:
    plt.yscale('symlog')

plt.title(args.title)
plt.legend(loc="lower right", title=args.ltitle)
if args.show:
    plt.show()
else:
    plt.savefig(f"{args.output}.png", dpi=300)
