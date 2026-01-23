from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
import matplotlib.pyplot as plt

DOMTree = xml.dom.minidom.parse("4groutes.xml")
routes = DOMTree.documentElement
flows = routes.getElementsByTagName("flow")


data = pd.DataFrame(columns=["id", "time", "vph"])


for flow in flows:
    flow_id = flow.getAttribute("id")[:2]
    flow_time = int(flow.getAttribute("begin"))
    vph = int(flow.getAttribute("vehsPerHour"))
    data.loc[len(data)] = [flow_id, flow_time, vph]


for i in range(0, 3600, 200):
    for j in ['f1', 'f2', 'f3', 'f4']:
        if i not in data[data['id'] == j]['time'].values:
            # print(data[data['id'] == j]['time'] == i)
            data.loc[len(data)] = [j, i, 0]


data.sort_values('time', inplace=True)
# print(data)
# 提取数据
f1_data = data[data['id'] == 'f1']
f2_data = data[data['id'] == 'f2']
f3_data = data[data['id'] == 'f3']
f4_data = data[data['id'] == 'f4']
# print(f1_data)

# 绘制阶梯图
plt.figure(figsize=(8, 6))

plt.step(f1_data['time'], f1_data['vph'],  where="post", label='f1', alpha=0.3, linewidth=4, color="red")
plt.step(f2_data['time'], f2_data['vph'],  where="post", label='f2', linestyle='dotted', color="green")
plt.step(f3_data['time'], f3_data['vph'],  where="post", label='f3', alpha=0.3, linewidth=4, color="purple")
plt.step(f4_data['time'], f4_data['vph'],  where="post", label='f4', linestyle='dotted', color="blue")

plt.xlabel('Simulation time (sec)')
plt.ylabel('Flow rate(veh/hr)')
# plt.title('Traffic flows vs simulation time within the traffic grid', loc='center')
plt.xticks(range(0, 2400, 200))
plt.xlim(0, 2400)
plt.legend()
# plt.grid(True)
plt.savefig('output_plot.png', dpi=300)
plt.show()
