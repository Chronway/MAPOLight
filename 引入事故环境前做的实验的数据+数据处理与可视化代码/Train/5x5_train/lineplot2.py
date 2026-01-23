import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


pr2pr = {
    '1': "100%",
    '0.2': "20%",
    '0.1': "10%",
    '0.05': "5%",
}

CUT_TIME = 11000000

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


ax = plt.gca()
sns.lineplot(data=data, x='Step', y='Value', hue='PR',  ax=ax)
ax.set_ylabel("Reward Value")
ax.get_legend().set_title("")

plt.savefig("5x5 Reward.pdf", dpi=300)
plt.show()