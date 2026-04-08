"""
generate_dataset.py — EpiModX training data construction pipeline
=================================================================
Implements the data construction described in the Methods section of:
  "Predicting disease-specific histone modifications and functional effects
   of non-coding variants by leveraging DNA language models"

Pipeline per histone mark:
  1. Load patient metadata (AD datasets.csv) to map accessions → disease groups
  2. Load all per-patient ChIP-seq BED files for the target histone
  3. Merge overlapping peaks (>80% reciprocal overlap) across patients
  4. Center each merged peak at its midpoint, extend to 4096 bp
  5. Label each region: binary vector (one entry per patient, 1 = has peak)
  6. Filter positives: ≥1 peak in central 2 kb OR >50% of seq length is peak
  7. Sample equal number of negatives (regions devoid of histone peaks),
     excluding ENCODE blacklist regions
  8. Write CSV: chrom, start, end, [patient_col × 22]

Output: Datasets/{histone}_generated.csv

Usage:
  python generate_dataset.py --histone H3K27ac
  python generate_dataset.py --histone H3K4me3 --blacklist hg38-blacklist.v2.bed.gz
  python generate_dataset.py --histone H3K27me3 --seed 42
"""

import os
import sys
import gzip
import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

# ── Constants ──────────────────────────────────────────────────────────────────

HISTONE_TYPES   = ["H3K27ac", "H3K4me3", "H3K27me3"]
SEQ_LENGTH      = 4096          # final window size centered on peak midpoint
CENTRAL_WINDOW  = 2000          # central 2 kb for positive-sample definition
OVERLAP_THRESH  = 0.80          # reciprocal overlap threshold for peak merging

# Disease group labels matching the AD datasets.csv (first column, forward-filled)
DISEASE_MAP = OrderedDict([
    ("No Cognitive Impairment",                    "NCI"),
    ("Mild Cognitive Impairment",                  "MCI"),
    ("Cognitive Impairment",                       "CI"),
    ("Alzheimer's Disease",                        "AD"),
    ("Alzheimer's Disease And Cognitive Impairment", "ADCI"),
])

# Chromosomes to include (autosomes + X, skip unplaced contigs)
VALID_CHROMS = {f"chr{i}" for i in list(range(1, 23)) + ["X"]}

# GRCh38 chromosome sizes (approximate) — used for negative sampling
# Source: https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh38
CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422,"chr11": 135086622,"chr12": 133275309,
    "chr13": 114364328,"chr14": 107043718,"chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX":  156040895,
}


# ── Helper functions ───────────────────────────────────────────────────────────

def load_metadata(csv_path: str):
    """
    Load and forward-fill disease group labels from AD datasets.csv.
    Returns a list of dicts with keys:
        group, group_abbrev, sample_id, DNase, CTCF, H3K27ac, H3K4me3, H3K27me3
    """
    df = pd.read_csv(csv_path)
    # Column 0 is the disease group (partially filled), column 1 is Sample ID
    df.columns = ["group", "sample_id", "years", "gender",
                  "DNase", "CTCF", "H3K27ac", "H3K4me3", "H3K27me3"]
    df["group"] = df["group"].ffill()  # forward-fill merged cells
    df["group_abbrev"] = df["group"].map(DISEASE_MAP)

    unmapped = df[df["group_abbrev"].isna()]["group"].unique()
    if len(unmapped):
        print(f"[WARN] Unmapped disease groups: {unmapped}")

    return df.dropna(subset=["group_abbrev"]).to_dict("records")


def load_bed_gz(path: str):
    """
    Load a gzipped BED file. Returns list of (chrom, start, end).
    Ignores comment/track lines and skips non-standard chromosomes.
    """
    peaks = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            if chrom in VALID_CHROMS:
                peaks.append((chrom, start, end))
    return peaks


def load_blacklist(path: str):
    """
    Load ENCODE blacklist BED (optionally gzipped).
    Returns dict: chrom → sorted list of (start, end) intervals.
    """
    if path is None or not os.path.exists(path):
        return {}
    bl = defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            bl[parts[0]].append((int(parts[1]), int(parts[2])))
    for chrom in bl:
        bl[chrom].sort()
    return dict(bl)


def overlaps_blacklist(chrom: str, start: int, end: int, blacklist: dict) -> bool:
    """Return True if [start, end) overlaps any blacklist region on chrom."""
    if chrom not in blacklist:
        return False
    for bl_start, bl_end in blacklist[chrom]:
        if bl_start >= end:
            break
        if bl_end > start:
            return True
    return False


def reciprocal_overlap(a_start, a_end, b_start, b_end) -> float:
    """Reciprocal overlap = intersection / min(len_a, len_b)."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    return inter / min(a_end - a_start, b_end - b_start)


def merge_peaks(all_peaks: list, overlap_thresh: float = OVERLAP_THRESH):
    """
    Merge peaks with >overlap_thresh reciprocal overlap across patients.
    Returns a list of (chrom, merged_start, merged_end).

    Strategy (greedy sweep per chromosome):
      Sort peaks by start. Greedily extend current cluster while any incoming
      peak has reciprocal overlap > threshold with the current merged interval.
    """
    # Group by chrom
    by_chrom = defaultdict(list)
    for chrom, start, end in all_peaks:
        by_chrom[chrom].append((start, end))

    merged = []
    for chrom in sorted(by_chrom):
        intervals = sorted(by_chrom[chrom])
        if not intervals:
            continue
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if reciprocal_overlap(cur_start, cur_end, start, end) >= overlap_thresh:
                # Merge: extend cluster to union
                cur_start = min(cur_start, start)
                cur_end   = max(cur_end, end)
            else:
                merged.append((chrom, cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((chrom, cur_start, cur_end))
    return merged


def center_and_extend(chrom, peak_start, peak_end, seq_length=SEQ_LENGTH, chrom_sizes=CHROM_SIZES):
    """
    Center on peak midpoint and extend to seq_length bp.
    Returns (chrom, new_start, new_end) or None if out of bounds.
    """
    mid = (peak_start + peak_end) // 2
    half = seq_length // 2
    new_start = mid - half
    new_end   = new_start + seq_length
    chrom_len = chrom_sizes.get(chrom, 0)
    if new_start < 0 or new_end > chrom_len:
        return None
    return chrom, new_start, new_end


def is_positive(chrom, region_start, region_end,
                patient_peaks_by_chrom: dict,
                seq_length=SEQ_LENGTH, central_window=CENTRAL_WINDOW):
    """
    Positive sample definition (paper Methods):
      (a) At least one patient peak in the central 2 kb region, OR
      (b) >50% of the sequence length is covered by peaks of any patient.

    Returns (is_positive, per_patient_labels_array).
    patient_peaks_by_chrom: dict[patient_idx] → dict[chrom] → sorted list of (start, end)
    """
    center_start = region_start + (seq_length - central_window) // 2
    center_end   = center_start + central_window

    labels = []
    has_central_peak = False
    total_coverage = 0  # nucleotides covered by any peak

    # Build a per-nucleotide coverage map (length = seq_length)
    coverage = np.zeros(seq_length, dtype=np.int8)

    for pat_idx, chrom_peaks in patient_peaks_by_chrom.items():
        pat_label = 0
        if chrom in chrom_peaks:
            for pk_start, pk_end in chrom_peaks[chrom]:
                if pk_end <= region_start:
                    continue  # peak is entirely before region — keep scanning
                if pk_start >= region_end:
                    break     # peaks are sorted; all remaining are past region
                # Peak overlaps [region_start, region_end)
                # Check if overlaps central window
                if pk_start < center_end and pk_end > center_start:
                    has_central_peak = True
                # Accumulate coverage
                cov_s = max(0, pk_start - region_start)
                cov_e = min(seq_length, pk_end - region_start)
                coverage[cov_s:cov_e] = 1
                pat_label = 1
        labels.append(pat_label)

    total_coverage = int(coverage.sum())
    frac_covered = total_coverage / seq_length

    positive = has_central_peak or (frac_covered > 0.5)
    return positive, labels


def sample_negatives(n_needed: int, all_peak_regions: set,
                     blacklist: dict, seq_length: int = SEQ_LENGTH,
                     rng: random.Random = None):
    """
    Randomly sample genomic regions devoid of histone peaks, excluding blacklist.
    all_peak_regions: set of (chrom, start, end) already used as positives.
    Returns list of (chrom, start, end).
    """
    if rng is None:
        rng = random.Random(42)

    # Build quick lookup: chrom → sorted list of occupied (start, end)
    occupied = defaultdict(list)
    for chrom, start, end in all_peak_regions:
        occupied[chrom].append((start, end))
    for chrom in occupied:
        occupied[chrom].sort()

    chroms = list(CHROM_SIZES.keys())
    chrom_weights = [CHROM_SIZES[c] for c in chroms]

    negatives = []
    max_attempts = n_needed * 50
    attempts = 0

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        chrom = rng.choices(chroms, weights=chrom_weights, k=1)[0]
        chrom_len = CHROM_SIZES[chrom]
        start = rng.randint(0, chrom_len - seq_length)
        end = start + seq_length

        if overlaps_blacklist(chrom, start, end, blacklist):
            continue

        # Check it doesn't overlap any known peak region
        overlaps = False
        for occ_start, occ_end in occupied.get(chrom, []):
            if occ_start >= end:
                break
            if occ_end > start:
                overlaps = True
                break
        if overlaps:
            continue

        negatives.append((chrom, start, end))

    if len(negatives) < n_needed:
        print(f"[WARN] Could only sample {len(negatives)}/{n_needed} negatives "
              f"after {attempts} attempts.")
    return negatives


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_dataset(histone: str, metadata_csv: str, bigwig_dir: str,
                  output_dir: str, blacklist_path: str = None,
                  seq_length: int = SEQ_LENGTH, seed: int = 42):
    """
    Full dataset construction pipeline for one histone mark.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Building dataset for: {histone}")
    print(f"{'='*60}")

    # ── 1. Load metadata ───────────────────────────────────────────────────────
    patients = load_metadata(metadata_csv)
    # Filter to patients that have a BED file for this histone
    patients_with_data = []
    for p in patients:
        accession = p.get(histone, "")
        bed_path = os.path.join(bigwig_dir, f"{histone}_{accession}.bed.gz")
        if os.path.exists(bed_path):
            p["bed_path"] = bed_path
            patients_with_data.append(p)
        else:
            print(f"[WARN] BED file not found for {p['sample_id']} / {histone}: {bed_path}")

    print(f"Patients with BED data: {len(patients_with_data)} / {len(patients)}")

    # Order patients by disease group to match task_dict order
    group_order = list(DISEASE_MAP.values())  # NCI, MCI, CI, AD, ADCI
    patients_with_data.sort(key=lambda p: (group_order.index(p["group_abbrev"]),
                                           p["sample_id"]))

    # Print group composition
    from collections import Counter
    group_counts = Counter(p["group_abbrev"] for p in patients_with_data)
    for g in group_order:
        print(f"  {g}: {group_counts.get(g, 0)} patients")

    # ── 2. Load all peaks ──────────────────────────────────────────────────────
    print("\nLoading BED files...")
    patient_peaks = {}      # pat_idx → dict[chrom] → sorted list of (start,end)
    all_peaks_flat = []     # (chrom, start, end) — all peaks from all patients

    for idx, p in enumerate(patients_with_data):
        peaks = load_bed_gz(p["bed_path"])
        by_chrom = defaultdict(list)
        for chrom, start, end in peaks:
            by_chrom[chrom].append((start, end))
        for chrom in by_chrom:
            by_chrom[chrom].sort()
        patient_peaks[idx] = dict(by_chrom)
        all_peaks_flat.extend(peaks)
        print(f"  [{idx+1:2d}/{len(patients_with_data)}] {p['sample_id']} "
              f"({p['group_abbrev']}): {len(peaks):,} peaks")

    # ── 3. Merge overlapping peaks ─────────────────────────────────────────────
    print(f"\nMerging {len(all_peaks_flat):,} peaks (overlap threshold={OVERLAP_THRESH})...")
    merged = merge_peaks(all_peaks_flat, OVERLAP_THRESH)
    print(f"  → {len(merged):,} merged peak regions")

    # ── 4. Center and extend each merged peak to seq_length ───────────────────
    print(f"\nCentering and extending to {seq_length} bp...")
    extended = []
    for chrom, peak_start, peak_end in merged:
        region = center_and_extend(chrom, peak_start, peak_end, seq_length)
        if region is not None:
            extended.append(region)
    print(f"  → {len(extended):,} valid regions after boundary check")

    # ── 5. Load blacklist ──────────────────────────────────────────────────────
    blacklist = load_blacklist(blacklist_path)
    if blacklist:
        print(f"\nBlacklist loaded: {sum(len(v) for v in blacklist.values()):,} regions")
    else:
        print("\nNo blacklist provided — negatives will not be blacklist-filtered")

    # ── 6. Assign per-patient labels; separate positives ──────────────────────
    print("\nAssigning labels and identifying positives...")
    positive_rows = []
    positive_regions = set()

    for chrom, start, end in extended:
        if overlaps_blacklist(chrom, start, end, blacklist):
            continue
        is_pos, labels = is_positive(chrom, start, end, patient_peaks, seq_length)
        if is_pos:
            positive_rows.append((chrom, start, end) + tuple(labels))
            positive_regions.add((chrom, start, end))

    print(f"  → {len(positive_rows):,} positive regions")

    # ── 7. Sample equal number of negatives ───────────────────────────────────
    n_neg = len(positive_rows)
    print(f"\nSampling {n_neg:,} negative regions...")
    neg_regions = sample_negatives(n_neg, positive_regions, blacklist, seq_length, rng)
    print(f"  → {len(neg_regions):,} negative regions sampled")

    # Build negative rows: all labels = 0
    n_patients = len(patients_with_data)
    negative_rows = [
        (chrom, start, end) + tuple([0] * n_patients)
        for chrom, start, end in neg_regions
    ]

    # ── 8. Build and write output CSV ──────────────────────────────────────────
    # Column names: chrom, start, end, then one column per patient
    patient_cols = [
        f"{p['group_abbrev']}_{p['sample_id']}" for p in patients_with_data
    ]
    columns = ["chrom", "start", "end"] + patient_cols

    all_rows = positive_rows + negative_rows
    df = pd.DataFrame(all_rows, columns=columns)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{histone}_generated.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Total rows : {len(df):,}  ({len(positive_rows):,} pos / {len(negative_rows):,} neg)")
    print(f"  Columns    : {len(columns)}  (3 coords + {n_patients} patient labels)")
    print(f"  Chroms     : {sorted(df['chrom'].unique())}")

    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EpiModX training dataset for one histone mark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--histone", type=str, required=True,
        choices=HISTONE_TYPES,
        help="Histone mark to process"
    )
    parser.add_argument(
        "--metadata", type=str, default="./Datasets/AD datasets.csv",
        help="Path to AD datasets.csv"
    )
    parser.add_argument(
        "--bigwig_dir", type=str, default="./Datasets/bigwig",
        help="Directory containing {HISTONE}_{ACCESSION}.bed.gz files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./Datasets",
        help="Directory to write {histone}_generated.csv"
    )
    parser.add_argument(
        "--blacklist", type=str, default=None,
        help="Path to ENCODE blacklist BED (optional, e.g. hg38-blacklist.v2.bed.gz). "
             "Download from: https://github.com/Boyle-Lab/Blacklist"
    )
    parser.add_argument(
        "--seq_length", type=int, default=SEQ_LENGTH,
        help="Sequence window size in bp"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative sampling"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        histone=args.histone,
        metadata_csv=args.metadata,
        bigwig_dir=args.bigwig_dir,
        output_dir=args.output_dir,
        blacklist_path=args.blacklist,
        seq_length=args.seq_length,
        seed=args.seed,
    )
