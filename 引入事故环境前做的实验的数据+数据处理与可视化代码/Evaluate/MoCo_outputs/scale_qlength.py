import os
import sys
import pandas as pd

for i in os.listdir(sys.argv[1]):
    if "csv" not in i:
        continue
    print(i)
    data = pd.read_csv(f"{sys.argv[1]}/{i}")
    data['total_stopped'] //= 3.17
    data['total_stopped'] = data['total_stopped'].astype(int)
    data.to_csv(f"{sys.argv[1]}/{i}", index=False)
    # print(data)