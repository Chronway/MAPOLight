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
parser.add_argument('--range', dest='range', help='', default=2000, type=int)
parser.add_argument('-a', '--arch', dest='arch', help='', default=['ppo', 'a3c', 'sac', "maxpressure", "uniform", "websters", "random"], nargs='+')
parser.add_argument('-p', '--path', dest='path', help='', default=["1.0", "0.2", "0.1", "0.05", "maxpressure", "uniform", "websters", "random"], nargs='+')

args = parser.parse_args()
print(args)


def read_bunch(path, arch, num):
    data = None
    for n in range(num):
        if data is None:
            data = pd.read_csv(f"{path}/{arch}_run{n}.csv")[['step_time', 'total_stopped', 'avg_wait_time', 'avg_speed']]
        else:
            _data = pd.read_csv(f"{path}/{arch}_run{n}.csv")[['step_time', 'total_stopped', 'avg_wait_time', 'avg_speed']]
            data = pd.concat([data, _data])
    data.insert(0, 'arch', arch)
    data.insert(0, 'path', path)
    return data


if __name__ == '__main__':
    tbl = pd.DataFrame(columns=['step_time', 'path', 'arch', 'total_stopped', 'avg_wait_time', 'avg_speed'])
    for a in args.arch:
        for p in args.path:
            try:
                tbl = pd.concat([tbl, read_bunch(p, a, args.num)])
            except FileNotFoundError:
                print(f"Can't find {a} in {p} !! ")


    tbl = tbl[tbl['step_time'] <= args.range]
    tbl.drop(columns=['step_time'], inplace=True)
    print(f"{'=' * 10} Temporal averages {'=' * 10}")
    print(tbl)
    tbl = tbl.groupby(["path", "arch"]).mean(numeric_only=False)
    tbl['total_stopped'] /= 16
    print(tbl)
    print(f"{'=' * 30}")

    # print(f"{'=' * 10} Temporal peaks (max) {'=' * 10}")
    # print(tbl.groupby(["path", "arch"]).max())
    # print(f"{'=' * 30}")

    # print(f"{'=' * 10} Temporal peaks (min) {'=' * 10}")
    # print(tbl.groupby(["path", "arch"]).min())
    # print(f"{'=' * 30}")
