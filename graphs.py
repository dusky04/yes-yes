from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader

from utils import download_dataset, unzip_files

CRICKET_EC_URL = "https://drive.google.com/file/d/1b1gKYveWSfAQB3S75Nq3t3MeiUYzDPkt/view?usp=drive_link"

download_dir = Path("zipped_data")
if not Path(download_dir).exists():
    download_dataset(download_dir, CRICKET_EC_URL)
    unzip_files(download_dir, "CricketEC")

dataset_dir = Path("CricketEC")
class_folders = list(dataset_dir.glob("*"))

our_classes = ["six", "four", "wicket", "catch"]
t = 0
for c in our_classes:
    t += len(list(Path(dataset_dir / c).rglob("*")))

video_paths = list(dataset_dir.rglob("*/*.avi"))
total_num_videos = len(video_paths)

print("Number of videos we made : ", t)
print("Videos in Cricshot10     :", total_num_videos - t)
print("Total number of videos   : ", total_num_videos)

classes = [class_folder.name for class_folder in class_folders]
df = pd.DataFrame(columns=["Total Time", "Average Clip Duration"], index=classes)

for class_path in class_folders:
    class_name = class_path.name
    print(f"Class: {class_name}")
    print("-" * 30)
    videos = class_path.glob("*.avi")
    video_lens = []
    for video in videos:
        vr = VideoReader(str(video))
        duration = len(vr) / vr.get_avg_fps()
        video_lens.append(duration)
    video_lens = np.array(video_lens)
    df.loc[class_name, "Total Time"] = video_lens.sum().round(2)
    df.loc[class_name, "Average Clip Duration"] = video_lens.mean().round(2)
    print(f"Maximum Video Length: {video_lens.max():.2f}")
    print(f"Minimum Video Length: {video_lens.min():.2f}")
    print(f"Average Video Length: {video_lens.mean():.2f}")
    print(f"Median  Video Length: {np.median(video_lens):.2f}")
    print("-" * 30)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "axes.labelweight": "bold",
    }
)

# ✅ Plot only after the loop is done
classes = df.index.tolist()
total_time = df["Total Time"].astype(float).values
avg_clip_duration = df["Average Clip Duration"].astype(float).values

x_pos = np.arange(len(classes))
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(16, 6))
color_total = "tab:blue"
ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
ax1.set_ylabel("Total Time (sec)", color=color_total, fontsize=12)
ax1.set_ylim(0, np.nanmax(total_time) * 1.2)
ax1.tick_params(axis="y", labelcolor=color_total)

ax1.bar(
    x_pos - bar_width / 2, total_time, bar_width, label="Total Time", color=color_total
)

ax2 = ax1.twinx()
color_avg = "tab:green"
ax2.set_ylabel("Average Clip Duration (sec)", color=color_avg, fontsize=12)
ax2.set_ylim(0, np.nanmax(avg_clip_duration) * 1.2)
ax2.tick_params(axis="y", labelcolor=color_avg)

ax2.bar(
    x_pos + bar_width / 2,
    avg_clip_duration,
    bar_width,
    label="Average Clip Duration",
    color=color_avg,
)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(classes, rotation=90, ha="center")
ax1.tick_params(axis="x", length=0)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2)

plt.title("Video Time Analysis by Class", fontsize=16, fontweight="bold")
fig.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


import matplotlib.pyplot as plt

# Data
labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
values = [15, 4, 5, 10, 6]

# Optional LaTeX style fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "axes.labelweight": "bold",
    }
)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=["royalblue", "lightgreen", "gold", "lightcoral", "plum"],
    wedgeprops={"edgecolor": "black", "linewidth": 1},
    textprops={"fontsize": 12},
)

# Equal aspect ratio ensures the pie is circular
ax.axis("equal")

# Title
ax.set_title(r"\textbf{Class Distribution}", fontsize=14, pad=15)

# Tight layout
plt.tight_layout()

# Save as PDF for LaTeX
plt.savefig("class_distribution.pdf", bbox_inches="tight")

plt.show()
