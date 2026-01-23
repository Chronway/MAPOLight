import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

fix = pl.read_csv("4x4_1_True_None_fix-time_None/run0.csv")
fix = fix.with_columns(
    desc = pl.lit("Fixed-time TLSC without accident")
).filter(
    pl.col('b3c3_sd') != -1
)

fixa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_fix-time_None/run0.csv")
fixa = fixa.with_columns(
    desc = pl.lit("Fixed-time TLSC with accident")
).filter(
    pl.col('b3c3_sd') != -1
)

ppo = pl.read_csv("4x4_1_True_None_ppo_None/run0.csv")
ppo = ppo.with_columns(
    desc = pl.lit("MARL TLSC without accident")
).filter(
    pl.col('b3c3_sd') != -1
)

ppoa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_ppo_None/run0.csv")
ppoa = ppoa.with_columns(
    desc = pl.lit("MARL TLSC with accident")
).filter(
    pl.col('b3c3_sd') != -1
)

maxp = pl.read_csv("4x4_1_True_None_maxp_None/run0.csv")
maxp = maxp.with_columns(
    desc = pl.lit("MaxPressure TLSC without accident")
).filter(
    pl.col('b3c3_sd') != -1
)

maxpa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_maxp_None/run0.csv")
maxpa = maxpa.with_columns(
    desc = pl.lit("MaxPressure TLSC with accident")
).filter(
    pl.col('b3c3_sd') != -1
)

# fix = fix.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)
# fixa = fixa.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)
# maxp = maxp.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)
# maxpa = maxpa.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)
# ppo = ppo.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)
# ppoa = ppoa.filter(600 <= pl.col("t")).filter(pl.col("t") < 1200)

d = pl.DataFrame([{
    "fixed_time": fix['b3c3_sd'].mean(),
    "maxpressure": maxp['b3c3_sd'].mean(),
    "ppo": ppo['b3c3_sd'].mean(),
},{
    "fixed_time": fixa['b3c3_sd'].mean(),
    "maxpressure": maxpa['b3c3_sd'].mean(),
    "ppo": ppoa['b3c3_sd'].mean(),
}])

print(d)
