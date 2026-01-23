from utils import *
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def compose_entry(path, e):
    spec = parse_dir_name(path)
    batch = read_batch(path)
    # batch = clip_time_step(batch, 2400)
    spec.update(calculate_accident_metric(batch))
    spec["edge"] = e
    return spec


l = []
for i in glob.glob("4x4_0.2_*B2C2*_None"):
    l.append(compose_entry(i, "D4D3_C3D3"))
for i in glob.glob("4x4_0.2_*None*_None"):
    l.append(compose_entry(i, "None"))
for i in glob.glob("4x4_1_*B2C2*_None"):
    l.append(compose_entry(i, "D4D3_C3D3"))
for i in glob.glob("4x4_1_*None*_None"):
    l.append(compose_entry(i, "None"))



df = pl.DataFrame(l)
C3D3 = df.filter((pl.col("edge").is_in(["None", "D4D3_C3D3"])))

def compose_data(data, metric):
    data = data.sort("pr", descending=True)
    data = data.sort("algorithm", descending=True)

    data = data.with_columns((pl.col("pr").cast(pl.Float32) * 100).cast(pl.Int32).cast(pl.String))
    data = data.with_columns(pl.when(pl.col("algorithm") == "ppo").then("MAPO-PPO/" + pl.col("pr") + "%").otherwise(pl.col("algorithm")).alias("algorithm"))
    # data = data.with_columns(pl.when(pl.col("algorithm") == "ppo").then(pl.lit("MARL TLSC")).otherwise(pl.col("algorithm")).alias("algorithm"))
    data = data.with_columns((pl.when(pl.col("algorithm") == "fix-time").then(pl.lit("fixed time")).otherwise(pl.col("algorithm"))).alias("algorithm"))
    data = data.with_columns((pl.when(pl.col("edge") == "None").then(pl.lit("Without accidents")).otherwise(pl.lit("With accidents"))).alias("edge"))
    
    print(data)
    return data

def savefig(data, name, metric):
    plt.clf()
    sns.barplot(compose_data(data, metric), x="algorithm", y=metric, hue="edge", edgecolor="0", width=0.6)
    plt.ylabel("Average speed (m/s)" if metric == "avg_speed" else "Average queue length (veh)")
    plt.legend().set_title("")
    plt.xlabel("")
    # plt.show()
    plt.savefig(name, dpi=300)

savefig(C3D3, "acc_avg_stopped.pdf", "avg_stopped")

savefig(C3D3, "acc_avg_speed.pdf", "avg_speed")