import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='bar2', description='柱状图2.0', epilog='')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=["websters"], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["def", "sat", "small", "tiny", "test", "test2", "test3", "test4"], nargs='+')
parser.add_argument('-t', '--type', dest='type', help='', default="avg_wait_time", choices=['avg_wait_time', 'avg_speed', 'total_stopped'])
parser.add_argument('-s', '--show', dest='show', action="store_true", help='', default=False)
parser.add_argument('--range', dest='range', type=int, help='', default=0)

args = parser.parse_args()

def create_df():
    # 创建DataFrame并设置各列的类型
    data = pd.DataFrame(columns=['arch', 'path', 'avg_wait_time', 'total_stopped', 'avg_speed'])
    data['arch'] = data['arch'].astype(str)
    data['path'] = data['path'].astype(str)
    data['avg_wait_time'] = data[args.type].astype(float)
    data['total_stopped'] = data[args.type].astype(float)
    data['avg_speed'] = data[args.type].astype(float)
    return data

def read_bunch(path, arch, num):
    # 在指定路径下读取一组文件，如ppo_run0.csv, ppo_run1.csv, ..., ppo_run9.csv
    # 并将结果拼接成一个DataFrame
    mean_data = create_df()
    for n in range(num):
        try:
            tmp_data = pd.read_csv(f"{path}/{arch}_real_run{n}.csv")
            if args.range != 0:  # 裁剪数据到指定模拟时长
                tmp_data = tmp_data[tmp_data["step_time"] <= args.range]
            tmp_data['arch'] = arch
            tmp_data['path'] = path
            mean_data.loc[len(mean_data)] = {"arch": arch, "path": path, "avg_wait_time": tmp_data["avg_wait_time"].mean(), "total_stopped": tmp_data["total_stopped"].mean(), "avg_speed": tmp_data["avg_speed"].mean()}
        except FileNotFoundError:
            pass
    return mean_data


mean_data = create_df()
for p in args.path:
    for a in args.arch:
        m_data = read_bunch(p, a, args.num)
        mean_data = pd.concat([mean_data, m_data], ignore_index=True)

print("load data finish.")

def post_process(data):
    # 数据后处理，给数据表添加便于画图的属性
    path2showname = {  # 将路径对应到显示名称
        "def": "Default",
        "sat": "Param1",
        "small": "Param2",
        "tiny": "Param3",
        "test": "Param4",
        "test2": "Param5",
        "test3": "Param6",
        "test4": "Param7",
    }

    data['showname'] = data['path'].apply(lambda x: path2showname.get(x, None))  # 将路径对应到显示名称
    data['total_stopped'] /= 8  # 将qlength指标平均到每个路口

post_process(mean_data)

print(mean_data)
data = mean_data


# 设置各类别颜色  
# palette_table = {
#     "def":      "#000000",
#     "sat":      "#000000",
#     "small":    "#000000",
#     "tiny":     "#000000",
#     "test":     "#000000",
#     "test2":    "#000000",
#     "test3":    "#000000",
#     "test4":    "#000000",
# }
palette_table = "Set3"  # 使用上面注释的代码可以自定义颜色，这里采用预设的配色方案

# 此行代码可以绘制箱图
# ax = sns.boxplot(data=data, x='showname', y=args.type, hue='showname', palette=palette_table, legend=False)

ax = sns.barplot(data=data, x='showname', y=args.type, hue='showname', palette=palette_table,
                err_kws={"linewidth": .5}, capsize=.3,  # 设置误差线属性
                legend=False)

ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_xticks(list(range(len(ax.get_xticklabels()))))  # 用于避免warning，可以去掉
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")  # 设置x坐标标签的对齐和旋转
ax.set_position([0.2, 0.3, 0.6, 0.6])  # 设置图像位置，以便x标签显示完整

# 根据不同的指标设置title
if args.type == 'avg_wait_time':
    ax.set_title('Average wait time (sec)')
elif args.type == 'avg_speed':
    ax.set_title('Average speed (m/s)')
elif args.type == 'total_stopped':
    ax.set_title('Average queue length (veh)')

if args.show:
    plt.show()
else:
    plt.savefig(f"boxplot.png", dpi=300)
