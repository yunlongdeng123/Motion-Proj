"""CPU-only orthographic diagnostic, without treating unknown shape as GT."""
import argparse
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("candidate", type=Path)
ns = ap.parse_args()
inputs = np.load(ns.candidate / "input_points_and_transform.npz", allow_pickle=False)
proposal = np.load(ns.candidate / "candidate_surface_samples.npz", allow_pickle=False)
real = inputs["points_native_m"]
generated = proposal["points_actor_native_m"]
size = inputs["size_lwh_m"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, (i, j), title in zip(axes, [(0, 2), (0, 1), (1, 2)],
                           ["Side: length / height", "Top: length / width", "Rear: width / height"]):
    ax.scatter(generated[::4, i], generated[::4, j], s=.4, c="#2878b5", alpha=.25,
               rasterized=True, label="Frozen learned proposal")
    ax.scatter(real[::3, i], real[::3, j], s=.4, c="#e77722", alpha=.3,
               rasterized=True, label="Original LiDAR observations")
    ax.plot([-size[i]/2, size[i]/2, size[i]/2, -size[i]/2, -size[i]/2],
            [-size[j]/2, -size[j]/2, size[j]/2, size[j]/2, -size[j]/2],
            color="#555555", linewidth=.8, linestyle="--")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("meters")
    ax.set_ylabel("meters")
    ax.grid(alpha=.2)
axes[0].legend(markerscale=5, fontsize=8, loc="lower left")
fig.suptitle("a6: frozen Omni proposal and partial original observations (unknown surfaces are not GT)", fontsize=11)
fig.savefig(ns.candidate / "prior_observation_fit.png", dpi=180)
plt.close(fig)
print(ns.candidate / "prior_observation_fit.png")
