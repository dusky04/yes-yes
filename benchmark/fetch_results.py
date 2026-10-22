"""
fetch_results.py — copy results from the GPU server and organise them by experiment.

Every run is pulled with rsync (incremental, so re-running only fetches what changed)
into a numbered folder per experiment, with one descriptively named folder per
configuration. Each experiment folder gets a README.md with its results table, a
results.csv and its charts; results/README.md is the index.

    results/
      README.md                                  index + headline numbers
      00_dataset_split/                          the fixed train/val/test assignment
      01_exp1_benchmark_A-E/run0_.. run3_../     the A-E benchmark, one folder per training recipe
      01_exp1_benchmark_A-E/exp1_test_eval/      Experiment 1: A-E test predictions + summary
      02_exp2_sampling_ablation/                 Experiment 2
      03_exp3_frame_sensitivity/                 Experiment 3
      04_exp4 .. 07_exp7/                        analysis experiments on the five A-E models
    each config folder: best.pt (weights), results.json, history.csv, train.log,
                        test_confusion_matrix.csv, test_predictions.csv

last.pt (optimiser state for resuming) stays on the server.

Run from the repo root (needs only matplotlib, not torch):
    uv run --no-project --with matplotlib python benchmark/fetch_results.py
    uv run --no-project --with matplotlib python benchmark/fetch_results.py --host root@<ip>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/home/yes-yes"
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]

# Validated categorical slots (dataviz reference palette, light mode) and chart ink
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# (remote dir, local section, local subfolder or None, title, what changed)
SOURCES = [
    ("runs_old_data", "01_exp1_benchmark_A-E", "run0_old-dataset_no-regularization",
     "Run 0: old 3,000-clip dataset, no regularization",
     "First run of the paper-faithful pipeline on the original Drive release (2,250/375/375). "
     "No augmentation, a single learning rate, no schedule."),
    ("runs", "01_exp1_benchmark_A-E", "run1_augment-labelsmooth-lowLR",
     "Run 1: 2,638-clip dataset + basic regularization",
     "New release with the fixed split. Adds clip-consistent augmentation (crop, flip, colour "
     "jitter), label smoothing 0.1, backbone at 0.1x learning rate, early stopping."),
    ("runs_v2", "01_exp1_benchmark_A-E", "run2_plus-temporal-jitter",
     "Run 2: + temporal jitter, random erasing, feature dropout",
     "Training clips get different frames every epoch; random erasing; dropout 0.2 on CNN features."),
    ("runs_v3", "01_exp1_benchmark_A-E", "run3_plus-cosine-ema-tta",
     "Run 3: + cosine LR schedule, weight EMA, test-time augmentation",
     "Current recipe. Warmup + cosine decay, EMA of the weights, TTA in the final evaluation."),
    ("experiments/exp2", "02_exp2_sampling_ablation", None,
     "Experiment 2: frame-sampling ablation",
     "ResNet-34, 16 frames; GRU and LSTM, each with uniform and pixel-intensity sampling. "
     "Everything else identical (batch 16, hidden 64, 40 epochs, lr 1e-3, wd 3e-3, seed 42), "
     "current regularization recipe (run 3)."),
    ("experiments/exp3", "03_exp3_frame_sensitivity", None,
     "Experiment 3: frame-count sensitivity",
     "ResNet-34 + GRU + uniform sampling at 8, 16, 24 and 32 frames; everything else identical "
     "to Experiment 2. Also measures single-clip inference latency and peak GPU memory."),
]

# Analysis experiments: (remote dir, local folder, title, CSVs rendered in the README)
ANALYSIS = [
    ("experiments/exp1_test_eval", "01_exp1_benchmark_A-E/exp1_test_eval",
     "Experiment 1: test-set evaluation of A-E", ["test_summary.csv"]),
    ("experiments/exp4_camera_perspective", "04_exp4_camera_perspective",
     "Experiment 4: camera-perspective robustness",
     ["camera_perspective_coverage.csv", "camera_perspective_results.csv"]),
    ("experiments/exp5_model_efficiency", "05_exp5_model_efficiency",
     "Experiment 5: computational efficiency and complexity", ["model_efficiency_results.csv"]),
    ("experiments/exp6_semantic_error_analysis", "06_exp6_semantic_error_analysis",
     "Experiment 6: semantic-layer and misclassification analysis",
     ["semantic_layer_results.csv", "class_confusion_analysis.csv"]),
    ("experiments/exp7_statistical_analysis", "07_exp7_statistical_analysis",
     "Experiment 7: statistical reliability",
     ["statistical_analysis_results.csv", "pairwise_significance_results.csv"]),
]
ANALYSIS_NOTE = (
    "Analysis of the five benchmark models A-E (`runs_v3`, current training recipe), produced by "
    "the scripts in `new-experiments/` through `new-experiments/run_analysis.sh`; Experiment 1 "
    "collects their per-clip test predictions, which 4, 6 and 7 analyse.\n")
PENDING_NOTE = "Not run yet: waits for all five A-E models to finish training (`runs_v3/`).\n"


def csv_to_markdown(path: Path, max_rows: int = 40) -> str:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    head, body = rows[0], rows[1:max_rows + 1]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    if len(rows) - 1 > max_rows:
        out.append(f"\n_{len(rows) - 1 - max_rows} more rows in `{path.name}`._")
    return "\n".join(out)


def fetch_analysis(host: str, out: Path) -> dict[str, bool]:
    cmd = f"cd {REMOTE_REPO} && for d in {' '.join(a[0] for a in ANALYSIS)}; do [ -d $d ] && echo $d; done"
    present = set(subprocess.run([*SSH, host, cmd], capture_output=True, text=True).stdout.split())
    found = {}
    for remote_dir, local, *_ in ANALYSIS:
        found[remote_dir] = remote_dir in present
        if found[remote_dir]:
            rsync(host, f"{REMOTE_REPO}/{remote_dir}/", out / local)
            subprocess.run(["rsync", "-a", "-e", " ".join(SSH),
                            f"{host}:{REMOTE_REPO}/{remote_dir}.log", str(out / local / "run.log")],
                           capture_output=True)
    return found


def write_analysis_readmes(out: Path) -> dict[str, str]:
    status = {}
    for remote_dir, local, title, tables in ANALYSIS:
        d = out / local
        d.mkdir(parents=True, exist_ok=True)
        files = sorted(p.name for p in d.iterdir() if p.name != "README.md")
        if not files:
            (d / "README.md").write_text(f"# {title}\n\n{PENDING_NOTE}")
            status[local] = "pending (waits for A-E)"
            continue
        parts = [f"# {title}\n", ANALYSIS_NOTE]
        for t in tables:
            if (d / t).exists():
                parts += [f"## `{t}`\n", csv_to_markdown(d / t) + "\n"]
        parts += [f"![{p}]({p})\n" for p in files if p.endswith(".png")]
        parts += ["## All files\n", "\n".join(f"- `{p}`" for p in files) + "\n"]
        (d / "README.md").write_text("\n".join(parts))
        status[local] = "done" if any(p.endswith(".csv") for p in files) else "incomplete"
    return status


def config_specs() -> dict[str, dict]:
    """Read the config table from paper_configs.py without importing torch."""
    text = (ROOT / "benchmark" / "paper_configs.py").read_text()
    pat = re.compile(r'"([\w-]+)": Config\("[\w-]+", "(\w+)",\s*"(\w+)",\s*"(\w+)",\s*(\d+),')
    return {m[1]: {"backbone": m[2], "rnn": m[3], "sampling": m[4], "frames": int(m[5])}
            for m in pat.finditer(text)}


def folder_name(cfg: str, spec: dict | None) -> str:
    if not spec:
        return cfg
    return f"{cfg}_{spec['backbone']}-{spec['rnn']}-{spec['sampling']}-{spec['frames']}f"


def remote_configs(host: str) -> dict[str, list[str]]:
    dirs = " ".join(s[0] for s in SOURCES)
    cmd = (f"cd {REMOTE_REPO} && for d in {dirs}; do [ -d $d ] || continue; "
           f"for c in $d/*/; do echo \"$d ${{c%/}}\"; done; done")
    out = subprocess.run([*SSH, host, cmd], capture_output=True, text=True, check=True).stdout
    found: dict[str, list[str]] = {}
    for line in out.splitlines():
        d, path = line.split(" ", 1)
        found.setdefault(d, []).append(path.rsplit("/", 1)[-1])
    return found


def rsync(host: str, src: str, dst: Path, *extra: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", *extra, "-e", " ".join(SSH), f"{host}:{src}", str(dst)],
                   check=True)


def fetch(host: str, out: Path, specs: dict) -> list[dict]:
    remote = remote_configs(host)
    sections = []
    for remote_dir, section, sub, title, desc in SOURCES:
        base = out / section / sub if sub else out / section
        configs = []
        for cfg in sorted(remote.get(remote_dir, [])):
            local = base / folder_name(cfg, specs.get(cfg))
            rsync(host, f"{REMOTE_REPO}/{remote_dir}/{cfg}/", local,
                  "--exclude", "last.pt", "--exclude", "epoch*.pt")
            subprocess.run(["rsync", "-a", "-e", " ".join(SSH),
                            f"{host}:{REMOTE_REPO}/{remote_dir}/{cfg}.log", str(local / "train.log")],
                           capture_output=True)  # older runs kept no per-config log
            configs.append((cfg, local))
        print(f"  {remote_dir:<18} -> {base.relative_to(out)}  ({len(configs)} configs)")
        sections.append({"remote": remote_dir, "section": section, "sub": sub, "title": title,
                         "desc": desc, "base": base, "configs": configs})
    rsync(host, f"{REMOTE_REPO}/data/CricketEC_split/split.json", out / "00_dataset_split" / "split.json")
    fetch_analysis(host, out)
    return sections


# --- reading results ----------------------------------------------------------
def progress(folder: Path) -> str:
    log = folder / "train.log"
    if log.exists():
        epochs = re.findall(r"epoch (\d+)/(\d+) \|", log.read_text(errors="replace"))
        if epochs:
            return f"running/stopped at epoch {epochs[-1][0]}/{epochs[-1][1]}"
    return "incomplete"


def row_for(cfg: str, folder: Path, specs: dict) -> dict:
    spec = specs.get(cfg, {})
    f = folder / "results.json"
    base = {"config": cfg, "folder": folder.name, "backbone": spec.get("backbone", ""),
            "rnn": spec.get("rnn", "").upper(), "sampling": spec.get("sampling", ""),
            "frames": spec.get("frames", "")}
    if not f.exists():
        return {**base, "status": progress(folder)}
    r = json.loads(f.read_text())
    c = r["config"]
    macro = r.get("test_macro_avg") or {
        k: sum(v[k] for v in r["test_per_class"].values()) / len(r["test_per_class"])
        for k in ("precision", "recall", "f1-score")}
    tta = r.get("test_tta") or {}
    cost = r.get("cost") or {}
    return {
        **base, "backbone": c["backbone"], "rnn": c["rnn"].upper(), "sampling": c["sampling"],
        "frames": c["num_frames"], "status": "done", "best_epoch": r["best_epoch"],
        "epochs": c["epochs"],
        "val_acc": r["val"]["accuracy"], "test_acc": r["test"]["accuracy"],
        "test_acc_tta": tta.get("accuracy"), "test_macro_precision": macro["precision"],
        "test_macro_recall": macro["recall"], "test_macro_f1": macro["f1-score"],
        "latency_ms": cost.get("latency_ms_per_clip"), "peak_mem_mb": cost.get("peak_gpu_mem_mb"),
        "train_minutes": r.get("train_minutes"), "test_sha": r["test_set"]["sha256"][:12],
        "per_class_source": r.get("test_per_class_source", "test"),
    }


def headline_acc(row: dict):
    return row.get("test_acc_tta") if row.get("test_acc_tta") is not None else row.get("test_acc")


# --- tables -------------------------------------------------------------------
def pct(v):
    return "" if v is None or v == "" else f"{100 * v:.2f}"


def num(v, d=4):
    return "" if v is None or v == "" else f"{v:.{d}f}"


def markdown_table(rows: list[dict], with_cost: bool) -> str:
    head = ["Config", "Model", "Sampling", "Frames", "Best epoch", "Val acc %", "Test acc %",
            "Test acc % (TTA)", "Macro P", "Macro R", "Macro F1"]
    if with_cost:
        head += ["Latency ms/clip", "Peak GPU MB"]
    head += ["Train min", "Status"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        done = r["status"] == "done"
        cells = [f"[{r['config']}](./{r['folder']}/)", f"{r['backbone']} + {r['rnn']}", r["sampling"],
                 str(r["frames"]),
                 f"{r['best_epoch']}/{r['epochs']}" if done else "",
                 pct(r.get("val_acc")), pct(r.get("test_acc")), pct(r.get("test_acc_tta")),
                 num(r.get("test_macro_precision")), num(r.get("test_macro_recall")),
                 num(r.get("test_macro_f1"))]
        if with_cost:
            cells += [num(r.get("latency_ms"), 2), num(r.get("peak_mem_mb"), 1)]
        cells += [num(r.get("train_minutes"), 1), r["status"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def same_test_set(rows: list[dict]) -> str:
    shas = {r["test_sha"] for r in rows if r.get("test_sha")}
    if not shas:
        return ""
    if len(shas) == 1:
        return f"All finished configs were scored on the same test set (fingerprint `{shas.pop()}`)."
    return f"**Warning: {len(shas)} different test sets in this folder.**"


# --- charts -------------------------------------------------------------------
def style(ax, ylabel: str, xlabel: str = "") -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)


def save(fig, path: Path) -> None:
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_exp2(rows: list[dict], path: Path) -> bool:
    by = {(r["rnn"], r["sampling"]): r for r in rows if r["status"] == "done"}
    if not by:
        return False
    rnns = ["GRU", "LSTM"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, label) in zip(axes, [("acc", "Test accuracy (%)"), ("test_macro_f1", "Test macro F1")]):
        for i, (samp, name, color) in enumerate([("uniform", "Uniform", BLUE),
                                                  ("pixel", "Pixel-intensity", ORANGE)]):
            xs, vals = [], []
            for j, rnn in enumerate(rnns):
                r = by.get((rnn, samp))
                if r is None:
                    continue
                v = 100 * headline_acc(r) if key == "acc" else r["test_macro_f1"]
                xs.append(j + (i - 0.5) * 0.3)
                vals.append(v)
            ax.bar(xs, vals, width=0.27, color=color, label=name, edgecolor="#fcfcfb", linewidth=2)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.1f}" if key == "acc" else f"{v:.3f}", ha="center",
                        va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(rnns)), [f"ResNet-34 + {r}" for r in rnns])
        ax.set_ylim(0, 100 if key == "acc" else 1.0)
        style(ax, label)
    axes[0].legend(frameon=False, loc="lower center", ncol=2, labelcolor=INK_2)
    fig.suptitle("Experiment 2: frame-sampling ablation (16 frames, test set)", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, path)
    return True


def chart_exp3(rows: list[dict], acc_path: Path, cost_path: Path) -> bool:
    done = sorted((r for r in rows if r["status"] == "done"), key=lambda r: r["frames"])
    if not done:
        return False
    frames = [r["frames"] for r in done]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    acc = [100 * headline_acc(r) for r in done]
    f1 = [100 * r["test_macro_f1"] for r in done]
    ax.plot(frames, acc, "-o", color=BLUE, lw=2, ms=7, label="Test accuracy",
            markeredgecolor="#fcfcfb", markeredgewidth=2)
    ax.plot(frames, f1, "-o", color=ORANGE, lw=2, ms=7, label="Test macro F1 (x100)",
            markeredgecolor="#fcfcfb", markeredgewidth=2)
    best = max(range(len(acc)), key=lambda i: acc[i])
    ax.annotate(f"{acc[best]:.1f}%", (frames[best], acc[best]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax.set_xticks(frames)
    lo = min(acc + f1)
    ax.set_ylim(max(0, lo - 8), min(100, max(acc + f1) + 5))
    style(ax, "Score (%)", "Sampled frames per clip")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    ax.set_title("Experiment 3: frame count vs accuracy (ResNet-34 + GRU, uniform)",
                 color=INK, fontsize=11, fontweight="bold")
    save(fig, acc_path)

    if any(r.get("latency_ms") is None for r in done):
        return True
    # two separate panels, not one dual-axis chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label, fmt in [(axes[0], "latency_ms", "Latency (ms per clip)", "{:.1f}"),
                                (axes[1], "peak_mem_mb", "Peak GPU memory (MB)", "{:.0f}")]:
        vals = [r[key] for r in done]
        ax.plot(frames, vals, "-o", color=BLUE, lw=2, ms=7, markeredgecolor="#fcfcfb",
                markeredgewidth=2)
        ax.annotate(fmt.format(vals[-1]), (frames[-1], vals[-1]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK)
        ax.set_xticks(frames)
        ax.set_ylim(0, max(vals) * 1.2)
        style(ax, label, "Sampled frames per clip")
    fig.suptitle("Experiment 3: frame count vs inference cost (batch size 1, A30)", color=INK,
                 fontsize=12, fontweight="bold")
    save(fig, cost_path)
    return True


def chart_benchmark(sections: list[dict], rows_by_section: dict, path: Path) -> bool:
    labels, vals = [], []
    for s in sections:
        if s["section"] != "01_exp1_benchmark_A-E":
            continue
        for r in rows_by_section[s["remote"]]:
            if r["config"] == "B" and r["status"] == "done":
                labels.append(s["title"].split(":")[0] + ("\n(TTA)" if r.get("test_acc_tta") else ""))
                vals.append(100 * headline_acc(r))
    if not vals:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(vals)), vals, width=0.5, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylim(0, 100)
    style(ax, "Test accuracy (%)")
    ax.set_title("Config B (ResNet-34 + GRU, 24 frames) across training recipes",
                 color=INK, fontsize=11, fontweight="bold")
    save(fig, path)
    return True


# --- READMEs ------------------------------------------------------------------
CONFIG_FILES = """Each config folder contains:
- `best.pt`: weights of the epoch with the best validation accuracy (EMA weights from run 3 on).
  Load with `CNNRNN` from `benchmark/paper_configs.py`; the checkpoint also stores the config and class order.
- `results.json`: all metrics, per-class precision/recall/F1, hyperparameters, test-set fingerprint
- `history.csv`: per-epoch train/val loss, accuracy and macro F1
- `train.log`: the training log
- `test_confusion_matrix.csv`, `test_predictions.csv`: per-clip test predictions
"""


def write_section_readme(folder: Path, title: str, desc: str, rows: list[dict], charts: list[str],
                         with_cost: bool) -> None:
    parts = [f"# {title}\n", desc + "\n",
             "Headline test accuracy uses test-time augmentation where it was enabled (decided "
             "before looking at test); the best epoch is always chosen on validation.\n",
             markdown_table(rows, with_cost) + "\n"]
    note = same_test_set(rows)
    if note:
        parts.append(note + "\n")
    parts += [f"![{c}]({c})\n" for c in charts]
    parts.append(CONFIG_FILES)
    (folder / "README.md").write_text("\n".join(parts))


def build_reports(out: Path, sections: list[dict], specs: dict) -> None:
    rows_by_section = {s["remote"]: [row_for(cfg, folder, specs) for cfg, folder in s["configs"]]
                       for s in sections}

    # 01: one README for all benchmark runs, one table per run
    bench = [s for s in sections if s["section"] == "01_exp1_benchmark_A-E"]
    bdir = out / "01_exp1_benchmark_A-E"
    bdir.mkdir(parents=True, exist_ok=True)
    charts = ["benchmark_progression.png"] if chart_benchmark(
        sections, rows_by_section, bdir / "benchmark_progression.png") else []
    parts = ["# Benchmark configurations A-E (paper Table 7)\n",
             "Each run folder is one training recipe; each builds on the one before. "
             "Config B was the reference while tuning the recipe.\n"]
    parts += [f"![{c}]({c})\n" for c in charts]
    all_bench = []
    for s in bench:
        rows = rows_by_section[s["remote"]]
        for r in rows:
            r["folder"] = f"{s['sub']}/{r['folder']}"
        all_bench += [{"run": s["sub"], **r} for r in rows]
        parts += [f"## [{s['title']}](./{s['sub']}/)\n", s["desc"] + "\n",
                  markdown_table(rows, with_cost=False) + "\n"]
    parts.append(CONFIG_FILES)
    (bdir / "README.md").write_text("\n".join(parts))
    if all_bench:
        write_csv(bdir / "results.csv", all_bench)

    # 02 / 03
    for s in sections:
        if s["section"] == "01_exp1_benchmark_A-E":
            continue
        rows = rows_by_section[s["remote"]]
        s["base"].mkdir(parents=True, exist_ok=True)
        charts = []
        if s["section"].startswith("02") and chart_exp2(rows, s["base"] / "sampling_ablation_comparison.png"):
            charts = ["sampling_ablation_comparison.png"]
        if s["section"].startswith("03") and chart_exp3(rows, s["base"] / "frame_count_vs_accuracy.png",
                                                         s["base"] / "frame_count_vs_cost.png"):
            charts = [c for c in ("frame_count_vs_accuracy.png", "frame_count_vs_cost.png")
                      if (s["base"] / c).exists()]
        write_section_readme(s["base"], s["title"], s["desc"], rows, charts,
                             with_cost=s["section"].startswith("03"))
        if rows:
            write_csv(s["base"] / "results.csv", rows)

    analysis_status = write_analysis_readmes(out)

    (out / "00_dataset_split" / "README.md").write_text(
        "# Dataset split\n\n`split.json` is the fixed train/val/test assignment of the 2,638-clip "
        "CricketEC release built by `benchmark/make_split.py` (seed 42): 15 duplicate files dropped, "
        "25 clips per class in val and test (375 each), 1,873 in train. Every run except run 0 uses it.\n")

    # index
    def best_line(remote):
        done = [r for r in rows_by_section.get(remote, []) if r["status"] == "done"]
        if not done:
            return "running" if rows_by_section.get(remote) else "not started"
        b = max(done, key=headline_acc)
        return f"best {b['config']}: {100 * headline_acc(b):.2f}% test acc ({len(done)}/{len(rows_by_section[remote])} configs done)"

    idx = ["# CricketEC results\n",
           "Everything copied back from the GPU server, organised by experiment. "
           "Regenerate with `uv run --no-project --with matplotlib python benchmark/fetch_results.py`.\n",
           "| Folder | What it is | Status |", "|---|---|---|",
           "| [00_dataset_split](./00_dataset_split/) | Fixed train/val/test split (2,638-clip release) | |"]
    for s in bench:
        idx.append(f"| [01_exp1_benchmark_A-E/{s['sub']}](./01_exp1_benchmark_A-E/{s['sub']}/) | {s['title']} | "
                   f"{best_line(s['remote'])} |")
    for s in sections:
        if s["section"] != "01_exp1_benchmark_A-E":
            idx.append(f"| [{s['section']}](./{s['section']}/) | {s['title']} | {best_line(s['remote'])} |")
    for _, local, title, _ in ANALYSIS:
        idx.append(f"| [{local}](./{local}/) | {title} | {analysis_status[local]} |")
    (out / "README.md").write_text("\n".join(idx) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="root@151.185.52.13")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--no-fetch", action="store_true", help="only rebuild READMEs/charts from local files")
    args = ap.parse_args()
    specs = config_specs()
    if args.no_fetch:
        sections = []
        for remote_dir, section, sub, title, desc in SOURCES:
            base = args.out / section / sub if sub else args.out / section
            configs = []
            if base.exists():
                for d in sorted(p for p in base.iterdir() if p.is_dir()):
                    configs.append((d.name.split("_", 1)[0], d))
            sections.append({"remote": remote_dir, "section": section, "sub": sub, "title": title,
                             "desc": desc, "base": base, "configs": configs})
    else:
        print(f"Fetching from {args.host}:{REMOTE_REPO}")
        sections = fetch(args.host, args.out, specs)
    build_reports(args.out, sections, specs)
    print(f"Organised results in {args.out} (start with README.md)")


if __name__ == "__main__":
    main()
