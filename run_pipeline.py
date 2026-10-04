"""
run_pipeline.py — run the whole dataset build in order.

    01 rename  ->  02 sort  ->  03 metadata  ->  04 split  ->  05 json  ->  06 csv

Stops at the first stage that needs something from you (e.g. the match mapping
in stage 04) with a clear message, so you fill that in and re-run. Re-running is
safe: each stage is idempotent.

Usage:
    python3 run_pipeline.py                 # run every stage
    python3 run_pipeline.py --from 03       # resume from a stage
    python3 run_pipeline.py --only 06       # run a single stage
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("01", "tools/01_rename_clips.py"),
    ("02", "tools/02_sort_into_folders.py"),
    ("03", "tools/03_extract_metadata.py"),
    ("04", "tools/04_make_split.py"),
    ("05", "tools/05_generate_annotations.py"),
    ("06", "tools/06_generate_taxonomy_csv.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="01")
    ap.add_argument("--only", dest="only")
    args = ap.parse_args()

    todo = [(n, s) for n, s in STAGES if (args.only == n) or
            (args.only is None and n >= args.start)]
    for name, script in todo:
        print(f"\n{'='*60}\n  STAGE {name}  {script}\n{'='*60}")
        r = subprocess.run([sys.executable, str(ROOT / script)])
        if r.returncode != 0:
            print(f"\n[pipeline] stage {name} stopped (exit {r.returncode}). "
                  f"Resolve the message above, then: "
                  f"python3 run_pipeline.py --from {name}")
            sys.exit(r.returncode)
    print("\n[pipeline] done.")


if __name__ == "__main__":
    main()
