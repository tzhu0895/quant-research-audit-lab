import numpy as np
from audit import moving_block_mean_ci, newey_west_mean_se, purged_split
rng=np.random.default_rng(42);x=rng.normal(0,.01,250)
print("SYNTHETIC DATA ONLY", moving_block_mean_ci(x))
print("HAC standard error:",newey_west_mean_se(x,5))
train,valid=purged_split(np.arange(30),np.arange(30)+3,20)
print("Purged split sizes:",len(train),len(valid))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3));ax.plot(np.cumsum(x));ax.set(title="Synthetic cumulative observations",xlabel="Observation",ylabel="Cumulative value");fig.tight_layout();fig.savefig("synthetic_report.png",dpi=120)
