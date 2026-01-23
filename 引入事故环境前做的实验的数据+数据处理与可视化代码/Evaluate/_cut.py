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
    
    try:
        df = df[df["step_time"] % 5 == 0]
        df = df[df['step_time'] >= 15]
        df = df.reset_index()
        df = df.drop(columns=["index"])
        if '4x4' in file_path:
            df = df[df['step_time'] <= 2000]
            df.to_csv(file_path, index=False)
        elif '5x5' in file_path:
            df = df[df['step_time'] < 3600]
            df.to_csv(file_path, index=False)
        elif 'MoCo' in file_path:
            df = df[df['step_time'] < 3600]
            df.to_csv(file_path, index=False)
    except KeyError:
        pass