import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re
from _collections import defaultdict
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as of
from plotly.subplots import make_subplots

def train_process():
    basic_cat = "D:\\Program\\python\\projects\\RL_signal\\output_v1\\output_dqn0.01\\"
    files = os.listdir(basic_cat)
    files.sort(key=lambda x: int(re.match('\D+(\d+)\.csv', x).group(1)))

    # store results
    mean_stop_value = []
    std_stop_value = []

    mean_avg_wait_time = []
    std_avg_wait_time = []

    for file in files:
        df = pd.read_csv(basic_cat + file)
        mean_stop = np.mean(df['reward'])
        std_stop = np.std(df['reward'])
        mean_stop_value.append(mean_stop)
        std_stop_value.append(std_stop)

        mean_wait = np.mean(df['avg_speed'])
        std_wait = np.std(df['avg_speed'])
        mean_avg_wait_time.append(mean_wait)
        std_avg_wait_time.append(std_wait)

    mean_value_stop = np.array(mean_stop_value)
    std_value_stop = np.array(std_stop_value)

    mean_value_wait = np.array(mean_avg_wait_time)
    std_value_wait = np.array(std_avg_wait_time)

    x = range(1, len(mean_value_stop) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, mean_value_stop)
    # ax.fill_between(x, mean_value_stop - std_value_stop, mean_value_stop + std_value_stop, alpha=0.2, label="total_stopped")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, mean_value_wait)
    ax.fill_between(x, mean_value_wait - std_value_wait, mean_value_wait + std_value_wait, alpha=0.2, label="avg_speed")
    plt.legend()
    plt.show()

def get_folder_data(folder_path):
    files = os.listdir(folder_path)
    files.sort(key=lambda x: int(re.match('\D+(\d+)\.csv', x).group(1)))

    ret = defaultdict(list)

    for file in files:
        df = pd.read_csv(folder_path + file)
        # queue length
        ret['mean_stop'].append(np.mean(df['total_stopped']))
        ret['std_stop'].append(np.std(df['total_stopped']))
        # avg_speed
        ret['mean_speed'].append(np.mean(df['avg_speed']))
        ret['std_speed'].append(np.std(df['avg_speed']))
        # avg_wait_time
        ret['mean_wait_time'].append(np.mean(df['avg_wait_time']))
        ret['std_wait_time'].append(np.std(df['avg_wait_time']))
        # reward
        ret['reward'].append(np.mean(df['reward']))

    for key in ret.keys():
        ret[key] = np.array(ret[key])
    return ret

def compare():
    basic_cat = "D:\\Program\\python\\projects\\RL_signal\\output_fix_demand\\DR20\\"
    # basic_cats = [basic_cat + i for i in ["output_pg0\\", "output_pg0.01\\", "output_pg0.03\\", "output_pg0.05\\", "output_pg0.1\\", "output_a3c0.01\\"]]
    basic_cats = [basic_cat + i for i in
                  ["output_pg0.02\\", "output_pg0.03\\", "output_pg0.05\\"]]
    labels = ["PR=0.02", "PR=0.03", "PR=0.05"]
    _, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    datas = []
    for cat in basic_cats:
        datas.append(get_folder_data(folder_path=cat))
    x = range(1, min([len(data['mean_wait_time']) for data in datas]) + 1)
    # queue length
    for data in datas:
        axes[0].plot(x, data['mean_stop'][0:len(x)])
        axes[0].fill_between(x, data['mean_stop'][0:len(x)] - data['std_stop'][0:len(x)],
                                data['mean_stop'][0:len(x)] + data['std_stop'][0:len(x)], alpha=0.3)

        axes[1].plot(x, data['mean_speed'][0:len(x)])
        axes[1].fill_between(x, data['mean_speed'][0:len(x)] - data['std_speed'][0:len(x)],
                                data['mean_speed'][0:len(x)] + data['std_speed'][0:len(x)], alpha=0.3)

        axes[2].plot(x, data['mean_wait_time'][0:len(x)])
        axes[2].fill_between(x, data['mean_wait_time'][0:len(x)] - data['std_wait_time'][0:len(x)],
                                data['mean_wait_time'][0:len(x)] + data['std_wait_time'][0:len(x)], alpha=0.3)

    for ax in axes:
        ax.legend(labels, fontsize=15)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Episode", fontsize=18)
    axes[0].set_title("Queue length", fontsize=18)
    axes[1].set_title("Average speed", fontsize=18)
    axes[2].set_title("Average waiting time", fontsize=18)

    axes[0].set_ylabel("Veh", fontsize=18)
    axes[1].set_ylabel("m/s", fontsize=18)
    axes[2].set_ylabel("s", fontsize=18)

    plt.show()


def plot_3d_compare():
    folders = "D:\\Program\\python\\projects\\RL_signal\\output_fix_demand\\"
    file_folders = [folders + i for i in ["DR10\\", "DR20\\", "DR30\\", "DR40\\", "DR50\\"]]
    values = []
    mean_mean_stop = []
    mean_mean_reward = []
    mean_mean_speed = []
    for file_group in file_folders:
        files = [file_group + j for j in ["output_pg0.01\\", "output_pg0.02\\", "output_pg0.03\\", "output_pg0.04\\", "output_pg0.05\\"]]
        value = []
        for folder in files:
            value.append(get_folder_data(folder))
        values.append(value)
        mean_mean_stop.append([np.mean(dic['mean_stop']) for dic in value])
        mean_mean_reward.append([np.mean(dic['reward']) for dic in value])
        mean_mean_speed.append([np.mean(dic['mean_speed']) for dic in value])

    DR = [10, 20, 30, 40, 50]
    PR = [1, 2, 3, 4, 5]

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mean_mean_reward, xticklabels=PR, yticklabels=DR, cmap=cmap, annot=True, fmt=".2f", linewidths=.5)
    plt.xlabel("PR (%)")
    plt.ylabel("DR (m)")
    plt.show()

    return mean_mean_speed

def compare_dqn():
    sns.set_style("whitegrid")
    basic_cat = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\"
    # basic_cats = [basic_cat + i for i in ["output_pg0\\", "output_pg0.01\\", "output_pg0.03\\", "output_pg0.05\\", "output_pg0.1\\", "output_a3c0.01\\"]]
    basic_cats = [basic_cat + i for i in
                  ["output_pg0.05_train\\", "output_dqn0.05_train\\"]]
    labels = ["PG","DQN"]
    _, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    datas = []
    for cat in basic_cats:
        datas.append(get_folder_data(folder_path=cat))
    x = range(1, min([len(data['mean_wait_time']) for data in datas]) + 1)
    # queue length
    for data in datas:
        axes[0].plot(x, data['mean_stop'][0:len(x)], linewidth=3)
        axes[0].fill_between(x, data['mean_stop'][0:len(x)] - data['std_stop'][0:len(x)],
                                data['mean_stop'][0:len(x)] + data['std_stop'][0:len(x)], alpha=0.3)

        axes[1].plot(x, data['mean_speed'][0:len(x)], linewidth=3)
        axes[1].fill_between(x, data['mean_speed'][0:len(x)] - data['std_speed'][0:len(x)],
                                data['mean_speed'][0:len(x)] + data['std_speed'][0:len(x)], alpha=0.3)

        axes[2].plot(x, data['mean_wait_time'][0:len(x)], linewidth=3)
        axes[2].fill_between(x, data['mean_wait_time'][0:len(x)] - data['std_wait_time'][0:len(x)],
                                data['mean_wait_time'][0:len(x)] + data['std_wait_time'][0:len(x)], alpha=0.3)

    for ax in axes:
        ax.legend(labels, fontsize=15)
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Episode", fontsize=18)
    axes[0].set_title("Queue length", fontsize=18)
    axes[1].set_title("Average speed", fontsize=18)
    axes[2].set_title("Average waiting time", fontsize=18)

    axes[0].set_ylabel("Veh", fontsize=18)
    axes[1].set_ylabel("m/s", fontsize=18)
    axes[2].set_ylabel("s", fontsize=18)


    plt.show()



def compare_radar():
    actuated_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_actuated\\"
    dqn_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_dqn0.05\\"
    pg_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_pg0.05\\"

    metrics = ["total_stop", "total_wait_time", "total_energy_consumption", "avg_speed"]
    signals = ["_signal1", "_signal2", "_signal3", "_signal4"]
    col_names = [i + j for i in metrics for j in signals]

    actuated = []
    dqn = []
    pg = []

    count = 1
    for method in [actuated_folder, dqn_folder, pg_folder]:
        method_files = os.listdir(method)
        df = pd.DataFrame()
        for file in [method_files[0]]:
            df = pd.read_csv(method + file)
        for file in method_files[1:]:
            df = pd.concat([df, pd.read_csv(method + file)], axis=0)
            df[df < 0] = 0
        if count == 1:
            actuated.append([np.mean(df[col_name]) for col_name in col_names])
        elif count == 2:
            dqn.append([np.mean(df[col_name]) for col_name in col_names])
        elif count == 3:
            pg.append([np.mean(df[col_name]) for col_name in col_names])
        count += 1

    print(actuated)
    print(dqn)
    print(pg)

    # normalize
    actuated = np.array(actuated[0])
    dqn = np.array(dqn[0])
    pg = np.array(pg[0])
    idxs = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    for idx in idxs:
        max_metric = np.max([actuated[idx], dqn[idx], pg[idx]])
        min_metric = np.min([actuated[idx], dqn[idx], pg[idx]])
        actuated[idx] = (actuated[idx] - min_metric) / (max_metric - min_metric)
        dqn[idx] = (dqn[idx] - min_metric) / (max_metric - min_metric)
        pg[idx] = (pg[idx] - min_metric) / (max_metric - min_metric)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Signal 1", "Signal 2", "Signal 3", "Signal 4"), specs=[[{"type": "polar"}, {"type": "polar"}], [{"type": "polar"}, {"type": "polar"}]])

    fig.add_trace(go.Scatterpolar(
        r=[actuated[0], actuated[4], actuated[8], actuated[12]],
        theta=metrics,
        fill='toself',
        name='ACS'
    ), row=1, col=1)
    fig.add_trace(go.Scatterpolar(
        r=[dqn[0], dqn[4], dqn[8], dqn[12]],
        theta=metrics,
        fill='toself',
        name='DQN'
    ), row=1, col=1)
    fig.add_trace(go.Scatterpolar(
        r=[pg[0], pg[4], pg[8], pg[12]],
        theta=metrics,
        fill='toself',
        name='PG'
    ), row=1, col=1)

    fig.add_trace(go.Scatterpolar(
        r=[actuated[1], actuated[5], actuated[9], actuated[13]],
        theta=metrics,
        fill='toself',
        name='ACS'
    ), row=1, col=2)
    fig.add_trace(go.Scatterpolar(
        r=[dqn[1], dqn[5], dqn[9], dqn[13]],
        theta=metrics,
        fill='toself',
        name='DQN'
    ), row=1, col=2)
    fig.add_trace(go.Scatterpolar(
        r=[pg[1], pg[5], pg[9], pg[13]],
        theta=metrics,
        fill='toself',
        name='PG'
    ), row=1, col=2)

    fig.add_trace(go.Scatterpolar(
        r=[actuated[2], actuated[6], actuated[10], actuated[14]],
        theta=metrics,
        fill='toself',
        name='ACS'
    ), row=2, col=1)
    fig.add_trace(go.Scatterpolar(
        r=[dqn[2], dqn[6], dqn[10], dqn[14]],
        theta=metrics,
        fill='toself',
        name='DQN'
    ), row=2, col=1)
    fig.add_trace(go.Scatterpolar(
        r=[pg[2], pg[6], pg[10], pg[14]],
        theta=metrics,
        fill='toself',
        name='PG'
    ), row=2, col=1)

    fig.add_trace(go.Scatterpolar(
        r=[actuated[3], actuated[7], actuated[10], actuated[15]],
        theta=metrics,
        fill='toself',
        name='ACS'
    ), row=2, col=2)
    fig.add_trace(go.Scatterpolar(
        r=[dqn[3], dqn[7], dqn[10], dqn[15]],
        theta=metrics,
        fill='toself',
        name='DQN'
    ), row=2, col=2)
    fig.add_trace(go.Scatterpolar(
        r=[pg[3], pg[7], pg[10], pg[15]],
        theta=metrics,
        fill='toself',
        name='PG'
    ), row=2, col=2)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    of.plot(fig)


    # fig.write_image("./fig.png")

def autolabel(rects, ax, decimal=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        if decimal:
            height = round(rect.get_height(), decimal)
        else:
            height = round(rect.get_height())
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def compare_bar():
    actuated_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_actuated\\"
    dqn_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_dqn0.05\\"
    pg_folder = "D:\\Program\\python\\projects\\RL_signal\\output_benchmark\\output_pg0.05\\"

    metrics = ["total_stop", "total_wait_time", "total_energy_consumption", "avg_speed"]
    signals = ["_signal1", "_signal2", "_signal3", "_signal4"]
    col_names = [i + j for i in metrics for j in signals]

    actuated = []
    dqn = []
    pg = []

    count = 1
    for method in [actuated_folder, dqn_folder, pg_folder]:
        method_files = os.listdir(method)
        df = pd.DataFrame()
        for file in [method_files[0]]:
            df = pd.read_csv(method + file)
        for file in method_files[1:]:
            df = pd.concat([df, pd.read_csv(method + file)], axis=0)
            df[df < 0] = 0
        if count == 1:
            actuated.append([np.mean(df[col_name]) for col_name in col_names])
        elif count == 2:
            dqn.append([np.mean(df[col_name]) for col_name in col_names])
        elif count == 3:
            pg.append([np.mean(df[col_name]) for col_name in col_names])
        count += 1

    _, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    labels = ["signal_1", "signal_2", "signal_3", "signal_4"]
    x = np.array([0, 2, 4, 6])
    width = 0.6
    # subplot 1: total_stop
    rect1 = axes[0][0].bar(x - width, actuated[0][0:4], width, label="ASC")
    rect2 = axes[0][0].bar(x, dqn[0][0:4], width, label="DQN")
    rect3 = axes[0][0].bar(x + width, pg[0][0:4], width, label="PG")
    autolabel(rect1, axes[0][0], decimal=1)
    autolabel(rect2, axes[0][0], decimal=1)
    autolabel(rect3, axes[0][0], decimal=1)
    axes[0][0].set_ylim([0, 90])
    axes[0][0].set_xlabel("(a)", fontsize=18)
    axes[0][0].set_ylabel("Queue Length (veh)")
    axes[0][0].set_xticks(x)
    axes[0][0].set_xticklabels(labels)

    # subplot 2: total_energy_consumption
    rect1 = axes[0][1].bar(x - width, actuated[0][4:8], width, label="ASC")
    rect2 = axes[0][1].bar(x, dqn[0][4:8], width, label="DQN")
    rect3 = axes[0][1].bar(x + width, pg[0][4:8], width, label="PG")
    autolabel(rect1, axes[0][1])
    autolabel(rect2, axes[0][1])
    autolabel(rect3, axes[0][1])
    axes[0][1].set_ylim([0, 12000])
    axes[0][1].set_xlabel("(b)", fontsize=18)
    axes[0][1].set_ylabel("Energy Consumption (Wh)")
    axes[0][1].set_xticks(x)
    axes[0][1].set_xticklabels(labels)

    # subplot 3: total_wait_time
    rect1 = axes[1][0].bar(x - width, actuated[0][8:12], width, label="ASC")
    rect2 = axes[1][0].bar(x, dqn[0][8:12], width, label="DQN")
    rect3 = axes[1][0].bar(x + width, pg[0][8:12], width, label="PG")
    autolabel(rect1, axes[1][0], decimal=2)
    autolabel(rect2, axes[1][0], decimal=2)
    autolabel(rect3, axes[1][0], decimal=2)
    axes[1][0].set_ylim([0, 35])
    axes[1][0].set_xlabel("(c)", fontsize=18)
    axes[1][0].set_ylabel("Waiting Time (s)")
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(labels)

    # subplot 4: avg_speed
    rect1 = axes[1][1].bar(x - width, actuated[0][12:16], width, label="ASC")
    rect2 = axes[1][1].bar(x, dqn[0][12:16], width, label="DQN")
    rect3 = axes[1][1].bar(x + width, pg[0][12:16], width, label="PG")
    autolabel(rect1, axes[1][1], decimal=2)
    autolabel(rect2, axes[1][1], decimal=2)
    autolabel(rect3, axes[1][1], decimal=2)
    axes[1][1].set_ylim([0, 2.7])
    axes[1][1].set_xlabel("(d)", fontsize=18)
    axes[1][1].set_ylabel("Average Speed (m/s)")
    axes[1][1].set_xticks(x)
    axes[1][1].set_xticklabels(labels)

    for i in [0, 1]:
        for j in [0, 1]:
            axes[i][j].legend()
    plt.show()



compare_dqn()