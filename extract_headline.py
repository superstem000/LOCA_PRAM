"""Extract the headline mid-K result: copies the K selected annotated PNGs
per (concentration, cycle) into <root>/analysis/results/ with names of the
form <conc>_<Nmin>_<ordinal>.png, and writes results.xlsx with two sheets:

  Sheet `tiles`         one row per copied file:  folder / name / counts
  Sheet `time_series`   the per_concentration_per_cycle_middleK<n>.csv data
                        (i.e. the numbers behind the mid-K over-time plot)

Runs against an already-completed analyze_assay.py --assays-root pass — it
reads each assay's analysis/per_tile.csv rather than re-running inference.

Usage:
    python extract_headline.py \\
        --assays-root "assays/ASFV/successful" \\
        --headline-k 15
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CONC_RE = re.compile(r"^(\d+\.\d+)")


def parse_concentration(name):
    m = _CONC_RE.match(os.path.basename(name))
    return m.group(1) if m else None


def middle_k(sub_df, k):
    """Take the k rows centered on the median of n_detections. Ties broken
    by original DataFrame order (stable sort)."""
    n = len(sub_df)
    if n <= k:
        return sub_df
    s = sub_df.sort_values("n_detections", kind="stable").reset_index(drop=True)
    start = (n - k) // 2
    return s.iloc[start:start + k]


def load_all_per_tile(assays_root, area):
    """Read per_tile.csv from every assay under assays_root that has one.
    Adds columns: assay (folder name), concentration, cycle_folder (parsed
    from the demo column). Recomputes density from n_detections / area so
    density is consistent across assays even if any per_tile.csv was written
    at a different --area."""
    rows_list = []
    for adir in sorted(glob.glob(os.path.join(assays_root, "*"))):
        if not os.path.isdir(adir):
            continue
        pt_path = os.path.join(adir, "analysis", "per_tile.csv")
        if not os.path.isfile(pt_path):
            continue
        conc = parse_concentration(os.path.basename(adir))
        if conc is None:
            continue
        df = pd.read_csv(pt_path)
        df["assay"] = os.path.basename(adir)
        df["assay_dir"] = os.path.abspath(adir)
        df["concentration"] = conc
        df["density"] = df["n_detections"] / area
        rows_list.append(df)
    if not rows_list:
        sys.exit(f"error: no analysis/per_tile.csv files found under {assays_root}")
    return pd.concat(rows_list, ignore_index=True)


_TILE_RE  = re.compile(r"tile_(\d+)_(\d+)_", re.IGNORECASE)
_CYCLE_RE = re.compile(r"(?:cycle|demo)_(\d+)", re.IGNORECASE)


def scan_annotated_pngs(assay_dir):
    """Walks <assay_dir>/analysis/<cycle_*/demo_*>/*.png and returns
       {(cycle_num, "row,col"): abs_path_to_png}
    Tile id is parsed from the PNG basename via the same tile_r_c_ regex
    analyze_assay.py used to build per_tile.csv, so keys line up 1-1 with
    per_tile.csv rows that used the parsed-from-filename tile source.
    """
    out = {}
    for png in glob.glob(os.path.join(assay_dir, "analysis", "*", "*.png")):
        cycle_folder = os.path.basename(os.path.dirname(png))
        m_cyc = _CYCLE_RE.match(cycle_folder)
        if not m_cyc:
            continue
        cyc = int(m_cyc.group(1))
        base = os.path.splitext(os.path.basename(png))[0]
        m_tile = _TILE_RE.match(base)
        if not m_tile:
            continue
        tile_id = f"{int(m_tile.group(1))},{int(m_tile.group(2))}"
        out[(cyc, tile_id)] = os.path.abspath(png)
    return out


def main(args):
    root = os.path.abspath(args.assays_root)
    if not os.path.isdir(root):
        sys.exit(f"error: --assays-root {root} is not a directory")

    K = int(args.headline_k)
    minutes_per_cycle = args.minutes_per_cycle

    results_dir = os.path.join(root, "analysis", "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"results dir: {results_dir}")

    # 1. Load all per_tile rows
    df = load_all_per_tile(root, args.area)
    print(f"loaded per_tile: {len(df)} rows from "
          f"{df['assay'].nunique()} assays / {df['concentration'].nunique()} concentrations")

    # 2. Pool by concentration, then take middle-K per cycle within each pool.
    #    Sort selected tiles by n_detections ASCENDING within each group —
    #    that becomes the ordinal 1..K used in the filename.
    selected = []
    for (conc, cycle), grp in df.groupby(["concentration", "cycle"]):
        mk = middle_k(grp, K).sort_values("n_detections", kind="stable").reset_index(drop=True)
        mk["ordinal"] = np.arange(1, len(mk) + 1)
        selected.append(mk)
    sel = pd.concat(selected, ignore_index=True)
    print(f"middle-{K} per (concentration, cycle) selected: {len(sel)} tiles")

    # 3. Build a per-assay lookup from (cycle_num, "row,col") -> PNG path.
    #    We scan the per-image annotated PNGs the analysis already wrote
    #    into <assay>/analysis/cycle_*/*.png, because per_tile.csv itself
    #    doesn't retain the source-image filename (it's grouped by tile).
    png_lookups = {a: scan_annotated_pngs(sel[sel["assay"] == a]["assay_dir"].iloc[0])
                   for a in sel["assay"].unique()}
    print(f"scanned annotated-PNG lookups for {len(png_lookups)} assays")

    # 4. Copy + rename each selected tile's annotated PNG.
    tile_rows = []
    for _, r in sel.iterrows():
        t_min = int(round((int(r["cycle"]) - 1) * minutes_per_cycle))
        new_name = f"{r['concentration']}_{t_min}min_{int(r['ordinal'])}.png"
        key = (int(r["cycle"]), str(r["tile"]))
        src = png_lookups[r["assay"]].get(key)
        if src is None:
            sys.exit(f"error: no annotated PNG found for assay={r['assay']!r} "
                     f"cycle={r['cycle']} tile={r['tile']!r} — was inference "
                     "run with --no-images, or is the tile using ordinal "
                     "fallback (unsupported)?")
        dst = os.path.join(results_dir, new_name)
        shutil.copyfile(src, dst)
        tile_rows.append({
            "folder": results_dir,
            "name":   new_name,
            "counts": int(round(float(r["n_detections"]))),
        })
    print(f"copied {len(tile_rows)} annotated PNGs into {results_dir}")

    tiles_df = pd.DataFrame(tile_rows)

    # 4. Load the time-series CSV that backs the mid-K graph
    ts_path = os.path.join(root, "analysis", f"per_concentration_per_cycle_middleK{K}.csv")
    if os.path.isfile(ts_path):
        ts_df = pd.read_csv(ts_path)
    else:
        # fall back: recompute a minimal time series from `sel`
        print(f"[note] {ts_path} not found — recomputing time series from per_tile data.")
        ts_df = (sel.groupby(["concentration", "cycle"], as_index=False)
                    .agg(n_tiles=("n_detections", "size"),
                         detections_mean=("n_detections", "mean"),
                         detections_std=("n_detections", "std"),
                         detections_median=("n_detections", "median"),
                         detections_sum=("n_detections", "sum"),
                         density_mean=("density", "mean"),
                         density_std=("density", "std")))
        ts_df["t_min"] = (ts_df["cycle"] - 1) * minutes_per_cycle
        ts_df = ts_df.sort_values(["concentration", "cycle"]).reset_index(drop=True)

    # 5. Write results.xlsx (two sheets)
    xlsx_path = os.path.join(results_dir, "results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        tiles_df.to_excel(w, sheet_name="tiles", index=False)
        ts_df.to_excel(w, sheet_name="time_series", index=False)
    print(f"wrote {xlsx_path}")
    print(f"  sheet `tiles`       : {len(tiles_df)} rows")
    print(f"  sheet `time_series` : {len(ts_df)} rows")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract the headline mid-K result into <root>/analysis/results/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--assays-root", required=True,
                   help="Parent folder that was passed to analyze_assay.py "
                        "--assays-root. Must contain per-assay analysis/ dirs "
                        "with per_tile.csv already written.")
    p.add_argument("--headline-k", type=int, default=15,
                   help="K value for the mid-K selection to extract.")
    p.add_argument("--area", type=float, default=1.2,
                   help="Per-tile FOV area — same value you passed to "
                        "analyze_assay.py (used to recompute density "
                        "consistently across assays).")
    p.add_argument("--minutes-per-cycle", type=float, default=5.0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
