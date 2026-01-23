from utils import *
import polars as pl
def compose_entry(path):
    spec = parse_dir_name(path)
    batch = read_batch(path)
    # batch = clip_time_step(batch, 2400)
    spec.update(calculate_accident_metric(batch))
    return spec


l = []
for i in glob.glob("4x4_*_True_*_None"):
    l.append(compose_entry(i))


pl.DataFrame(l).write_csv("table.csv")