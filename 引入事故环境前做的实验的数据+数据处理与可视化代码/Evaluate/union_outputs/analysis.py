# usage: python analysis2.py 
# dependence: pandas==2.1.1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                 epilog='Text at the bottom of help')

parser.add_argument('-n', '--num', dest='num', help='', default=10, type=int)
parser.add_argument('--range', dest='range', help='', default=3600, type=int)
parser.add_argument('--net', dest='net', help='', default=['4x4', '5x5', 'MoCo'], nargs='+')
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', 'uniform', "maxpressure", 'websters', "random"], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.2", "0.1", "0.05", 'uniform', "maxpressure", 'websters', "random"], nargs='+')

args = parser.parse_args()
print(args)


def read_bunch(net, path, arch, num):
    data = None
    count = 0
    for n in range(num):
        print('read', f"{net}/{path}/{arch}_run{n}.csv")
        try:
            if data is None:
                data = pd.read_csv(f"{net}/{path}/{arch}_run{n}.csv")[['step_time', 'total_stopped', 'avg_wait_time', 'avg_speed']]
            else:
                _data = pd.read_csv(f"{net}/{path}/{arch}_run{n}.csv")[['step_time', 'total_stopped', 'avg_wait_time', 'avg_speed']]
                data = pd.concat([data, _data])
            count += 1
        except FileNotFoundError:
            pass
    if data is None:
        return None
    data.insert(0, 'net', net)
    data.insert(0, 'arch', arch)
    data.insert(0, 'path', path)
    print(f"{net}/{path}/{arch}:{count}")
    return data


if __name__ == '__main__':
    tbl = pd.DataFrame(columns=['step_time', 'net', 'path', 'arch', 'total_stopped', 'avg_wait_time', 'avg_speed'])
    for n in args.net:
        for p in args.path:
            for a in args.arch:
                if (d := read_bunch(n, p, a, args.num)) is not None:
                    tbl = pd.concat([tbl, d])


    tbl = tbl[tbl['step_time'] <= args.range]
    tbl.drop(columns=['step_time'], inplace=True)

    tbl.loc[tbl['net'] == '4x4', 'total_stopped'] /= 16
    tbl.loc[tbl['net'] == '5x5', 'total_stopped'] /= 25
    tbl.loc[tbl['net'] == 'MoCo', 'total_stopped'] /= 8
    
    print(f"{'=' * 10} Temporal averages {'=' * 10}")
    tbl = tbl.groupby(["net", "path", "arch"]).mean(numeric_only=False)
    print(tbl)
    print(f"{'=' * 30}")

    # print(f"{'=' * 10} Temporal peaks (max) {'=' * 10}")
    # print(tbl.groupby(["path", "arch"]).max())
    # print(f"{'=' * 30}")

    # print(f"{'=' * 10} Temporal peaks (min) {'=' * 10}")
    # print(tbl.groupby(["path", "arch"]).min())
    # print(f"{'=' * 30}")
