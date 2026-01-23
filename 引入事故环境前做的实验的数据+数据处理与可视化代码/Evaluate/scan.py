import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('root', help='搜索的根目录', default=os.getcwd(), nargs="?")
args = parser.parse_args()

# 遍历搜索csv文件
def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.relpath(os.path.join(root, file), os.getcwd()))  # 以当前工作目录为相对路径
    return csv_files

csv_files = find_csv_files(args.root)

for file_path in csv_files:
    df = pd.read_csv(file_path)
    print(f"{file_path}::\t\t", end="", flush=True)
    try:
        print(f"{df['step_time'].max()}::{len(df)}")
    except KeyError:
        print("no_step_time")