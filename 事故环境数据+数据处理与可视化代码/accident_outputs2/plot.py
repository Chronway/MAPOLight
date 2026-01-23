import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

fix = pl.read_csv("4x4_1_True_None_fix-time_None/run0.csv")
fix = fix.with_columns(
    desc = pl.lit("Without accident")
)
# data2 = pl.read_csv("4x4_1_False_None_fix-time_None1/run0.csv")
# data2 = data2.with_columns(
#     desc = pl.lit("fn2")
# )
fix = fix.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))

print(fix["sd"].mean())
fixa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_fix-time_None/run0.csv")
fixa = fixa.with_columns(
    desc = pl.lit("With accident")
)
fixa = fixa.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))

ppo = pl.read_csv("4x4_1_True_None_ppo_None/run0.csv")
ppo = ppo.with_columns(
    desc = pl.lit("Without accident")
)
ppo = ppo.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))
print(ppo["sd"].mean())
ppoa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_ppo_None/run0.csv")
ppoa = ppoa.with_columns(
    desc = pl.lit("With accident")
)
ppoa = ppoa.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))

maxp = pl.read_csv("4x4_1_True_None_maxp_None/run0.csv")
maxp = maxp.with_columns(
    desc = pl.lit("Without accident")
)
maxp = maxp.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))

maxpa = pl.read_csv("4x4_1_True_B3C3_B3C3_B3C3_maxp_None/run0.csv")
maxpa = maxpa.with_columns(
    desc = pl.lit("With accident")
)
maxpa = maxpa.with_columns(b3c3_sd=pl.col("b3c3_sd").rolling_mean(5, min_samples=1))

# data = pl.concat([ data3, data4])
fix = pl.concat([fix, fixa])

fix = fix.filter(
    pl.col('sd') != -1
)

plt.figure(figsize=(10, 4))

ax = sns.lineplot(fix, x="t", y='b3c3_sd', hue="desc")
ax.axvspan(700, 1900, color='red', alpha=0.14, label='Duration of the accident')

plt.legend(loc="upper right").set_title("")
plt.xlabel("Simulation Time")
plt.ylabel("Average speed (m/s)")
# plt.show()
plt.savefig("fixedtime_acc.pdf", dpi=300)
