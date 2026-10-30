"""
match_release.py — map the final annotated CricketEC release onto the split the
models were trained on, by file content.

The final release (Aryan's CricketEC_release_final) renames clips and has its own
train/val/test split; its annotations (annotations/*_split_match_disjoint.json) are
keyed by those new names. Our models were trained on data/CricketEC_split. Matching
the two by MD5 tells us
  * whether every clip we trained/tested on is in the release (and vice versa),
  * how the release's split lines up with ours (a cross-tab),
  * the camera perspective of each of our clips, and so whether every perspective
    is present in our train, val and test,
and writes the per-clip perspective labels Experiment 4 reads.

Run from the repo root:
    python3 benchmark/match_release.py
Outputs (into --out, default results/04_exp4_camera_perspective/release_match/):
    clip_mapping.csv           every clip of ours with its release counterpart + annotations
    perspective_coverage.csv   our split x (layer, perspective) clip counts
    exp4_annotations.csv       file_path + semantic_layer + camera_perspective for Experiment 4
    summary.txt                the numbers below, as text
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index(root: Path) -> dict[str, list[dict]]:
    """md5 -> clips under root/<split>/.../<class>/<name>.avi"""
    out = defaultdict(list)
    for p in sorted(root.rglob("*.avi")):
        rel = p.relative_to(root)
        out[md5(p)].append({"split": rel.parts[0], "class": p.parent.name.lower(),
                            "name": p.name, "rel": rel.as_posix()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "data" / "CricketEC_split")
    ap.add_argument("--release", type=Path, default=ROOT / "data" / "release_final")
    ap.add_argument("--annotations", type=Path, default=ROOT / "annotations")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "04_exp4_camera_perspective" / "release_match")
    args = ap.parse_args()

    ann, ann_split = {}, {}
    for s in SPLITS:
        for c in json.load(open(args.annotations / f"{s}_split_match_disjoint.json"))["clips"]:
            name = Path(c["file_path"]).name
            ann[name], ann_split[name] = c, s

    print("hashing our split ...")
    ours = index(args.ours)
    print("hashing the release ...")
    rel = index(args.release)

    lines = []
    def say(t=""):
        print(t); lines.append(t)

    n_ours = sum(len(v) for v in ours.values())
    n_rel = sum(len(v) for v in rel.values())
    say(f"our split: {n_ours} clips | release: {n_rel} clips | annotations: {len(ann)} clips")

    rows, cross, coverage = [], Counter(), Counter()
    unmatched_ours, class_mismatch = [], []
    for h, clips in ours.items():
        r = rel.get(h, [])
        for c in clips:
            if not r:
                unmatched_ours.append(c["rel"])
                continue
            rc = r[0]
            a = ann.get(rc["name"], {})
            persp = (a.get("camera_perspective") or "").strip()
            layer = a.get("semantic_layer") or ""
            if rc["class"] != c["class"]:
                class_mismatch.append(f"{c['rel']} -> {rc['rel']}")
            cross[(c["split"], rc["split"])] += 1
            coverage[(c["split"], layer, persp or "<missing>")] += 1
            rows.append({"our_split": c["split"], "file_path": f"{c['class']}/{c['name']}",
                         "release_split": rc["split"], "release_file": rc["rel"],
                         "annotation_split": ann_split.get(rc["name"], ""),
                         "class": c["class"], "semantic_layer": layer, "camera_perspective": persp,
                         "chirality": a.get("chirality") or "", "release_copies": len(r)})
    our_hashes = set(ours)
    unmatched_rel = [c["rel"] for h, cs in rel.items() if h not in our_hashes for c in cs]

    say(f"matched by content: {len(rows)} of our {n_ours} clips")
    say(f"our clips not in the release: {len(unmatched_ours)} {unmatched_ours[:8]}")
    say(f"release clips not in our split: {len(unmatched_rel)} {unmatched_rel[:8]}")
    say(f"class disagreements: {len(class_mismatch)} {class_mismatch[:5]}")
    say("\nour split (rows) vs release split (columns):")
    say("        " + "".join(f"{s:>8}" for s in SPLITS))
    for s in SPLITS:
        say(f"{s:<8}" + "".join(f"{cross[(s, t)]:>8}" for t in SPLITS))
    same = sum(cross[(s, s)] for s in SPLITS)
    say(f"same split on both sides: {same} of {len(rows)}")

    persps = sorted({(l, p) for _, l, p in coverage})
    say("\ncamera perspective coverage in OUR split:")
    say(f"{'layer':<9}{'camera_perspective':<30}" + "".join(f"{s:>7}" for s in SPLITS))
    cov_rows = []
    for l, p in persps:
        cnt = [coverage[(s, l, p)] for s in SPLITS]
        flag = "" if all(cnt) or p == "<missing>" else "   <- not in every split"
        say(f"{l:<9}{p:<30}" + "".join(f"{n:>7}" for n in cnt) + flag)
        cov_rows.append({"semantic_layer": l, "camera_perspective": p, **dict(zip(SPLITS, cnt))})

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("clip_mapping.csv", rows), ("perspective_coverage.csv", cov_rows)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    with open(args.out / "exp4_annotations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "semantic_layer", "camera_perspective"])
        w.writeheader()
        w.writerows({"file_path": r["file_path"], "semantic_layer": r["semantic_layer"],
                     "camera_perspective": r["camera_perspective"]} for r in rows)
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
