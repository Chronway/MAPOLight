import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


pr2pr = {
    '1': "100%",
    '0.2': "20%",
    '0.1': "10%",
    '0.05': "5%",
}

CUT_TIME = 1400000

data = pd.DataFrame(columns=['Wall time', 'Step', 'Value', 'PR'])
data['Wall time'] = data['Wall time'].astype(float)
data['Step'] = data['Step'].astype(float)
data['Value'] = data['Value'].astype(float)
data['PR'] = data['PR'].astype(str)
for pr in ['1', '0.2', '0.1', '0.05']:
    fp = "PPO_" + pr + ".csv"
    df = pd.read_csv(fp)
    df = df[df['Step'] < CUT_TIME]
    df['PR'] = 'MAPO-PPO/' + pr2pr[pr]
    data = pd.concat([data, df])

    fp = "A3C_" + pr + ".csv"
    df = pd.read_csv(fp)
    df = df[df['Step'] < CUT_TIME]
    df['PR'] = 'MAPO-A3C/' + pr2pr[pr]
    data = pd.concat([data, df])

    fp = "SAC_" + pr + ".csv"
    df = pd.read_csv(fp)
    df = df[df['Step'] < CUT_TIME]
    df['PR'] = 'MAPO-SAC/' + pr2pr[pr]
    data = pd.concat([data, df])

    # df = pd.read_csv("random.csv")
    # df = df[df['Step'] < CUT_TIME]
    # df['PR'] = 'Random'
    # data = pd.concat([data, df])


ax = plt.gca()
sns.lineplot(data=data[data['PR'].str.contains('100%')], x='Step', y='Value', hue='PR',  ax=ax)
sns.lineplot(data=data[data['PR'].str.contains('10%')], x='Step', y='Value', hue='PR', linestyle='dotted', ax=ax)
sns.lineplot(data=data[data['PR'].str.contains('5%')], x='Step', y='Value', hue='PR', linestyle='dashed', ax=ax)
# sns.lineplot(data=data[data['PR'] == "Random"], x='Step', y='Value', hue='PR', palette={"red"}, ax=ax)
plt.axhline(-575, xmin=0, xmax=1.4e6, color="red", label="random")
plt.axhline(-460, xmin=0, xmax=1.4e6, color="#EF827F", label="websters")
plt.axhline(-440, xmin=0, xmax=1.4e6, color="#9A3335", label="maxpressure")

plt.legend()
ax.set_ylabel("Reward Value")
ax.get_legend().set_title("")

# plt.legend()
plt.savefig("4x4 Reward.pdf", dpi=300)
plt.show()
