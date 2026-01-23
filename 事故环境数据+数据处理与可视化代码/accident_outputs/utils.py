import polars as pl  # 5~10x faster than pandas
import glob

def parse_dir_name(name: str):
    segments = name.split("_")
    return {
        "net": segments[0],
        "pr": segments[1],
        "collaborate": True if segments[2] == "True" else False,
        "accident_edge": ','.join(segments[3:-2]),
        "algorithm": segments[-2],
        "extra_module": segments[-1]
    }

def read_batch(directory: str):
    return pl.read_csv(f"{glob.escape(directory)}/*.csv")  # polars支持使用通配符批量读取，比pandas先读取再连接方便

def clip_time_step(df, limit: int):
    return df.filter(pl.col("step_time") <= limit)

def calculate_metric(df):
    return {
        "avg_speed": df["avg_speed"].mean(),
        "avg_stopped": df["avg_stopped"].mean(),
    }

def calculate_accident_metric(df):
    return {
        "C2_avg_speed": df["C2_avg_speed"].mean(),
        "C2_avg_stopped": df["C2_avg_stopped"].mean(),
        "avg_speed": df["avg_speed"].mean(),
        "avg_stopped": df["avg_stopped"].mean(),
    }