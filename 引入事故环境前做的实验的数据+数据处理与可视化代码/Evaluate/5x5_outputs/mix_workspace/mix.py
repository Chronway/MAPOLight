import pandas as pd
import os

for _, _, files in os.walk("old"):
    for f in files:
        old_fp = "old/" + f
        new_fp = "new/" + f
        old_df = pd.read_csv(old_fp)
        new_df = pd.read_csv(new_fp)
        old_df['avg_wait_time'] = new_df['avg_wait_time']
        os.makedirs("out", exist_ok=True)
        old_df.to_csv(f"out/{f}", index=None)