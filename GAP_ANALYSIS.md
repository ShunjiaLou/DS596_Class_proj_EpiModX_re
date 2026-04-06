# EpiModX — Gap Analysis & Fix Log

**Paper**: Predicting disease-specific histone modifications and functional effects of non-coding variants by leveraging DNA language models  
**Repo**: github.com/yanwu20/EpiModX  
**Analysis date**: 2026-04-06  
**Last updated**: 2026-04-06

---

## Data Understanding

### AD datasets.csv structure
| Column | Content |
|--------|---------|
| (unnamed col 0) | Disease group (forward-filled): NCI, MCI, CI, AD, AD+CI |
| Sample ID | ENCODE donor accession |
| Years | Age |
| Gender | Sex |
| DNase | ENCODE file accession for DNase-seq peak BED |
| CTCF | ENCODE file accession for CTCF ChIP-seq peak BED |
| H3K27ac | ENCODE file accession for H3K27ac peak BED |
| H3K4me3 | ENCODE file accession for H3K4me3 peak BED |
| H3K27me3 | ENCODE file accession for H3K27me3 peak BED |

### Patient counts per group (matches task_dict exactly)
| Group | Abbrev | task_dict key | Count |
|-------|--------|---------------|-------|
| No Cognitive Impairment | NCI | task1 | 6 |
| Mild Cognitive Impairment | MCI | task2 | 5 |
| Cognitive Impairment | CI | task3 | 4 |
| Alzheimer's Disease | AD | task4 | 4 |
| Alzheimer's Disease And Cognitive Impairment | ADCI | task5 | 3 |
| **Total** | | | **22** |

### Label interpretation
Each training CSV (`{histone}_all_data.csv`) has columns:  
`chrom, start, end, [NCI_p1, ..., NCI_p6, MCI_p1, ..., MCI_p5, CI_p1, ..., CI_p4, AD_p1, ..., AD_p4, ADCI_p1, ..., ADCI_p3]`  
= 22 binary labels per genomic region, one per patient.

### BED file naming convention
`Datasets/bigwig/{HISTONE}_{ENCODE_ACCESSION}.bed.gz`  
e.g. `H3K27ac_ENCFF721ZGP.bed.gz`

### Generated dataset sizes
| Histone | Total rows | Positives | Negatives | File size |
|---------|-----------|-----------|-----------|-----------|
| H3K27ac | 434,110 | 217,055 | 217,055 | ~28 MB |
| H3K4me3 | 135,024 | 67,512 | 67,512 | ~8.7 MB |
| H3K27me3 | 623,356 | 311,678 | 311,678 | ~41 MB |

---

## Issues Found

### CRITICAL — Blockers (prevent any training)

#### [C1] `parallel_experts` package not installed
- **File**: `utils/Moe.py` line 6
- **Import**: `from parallel_experts import RandomMoE, TaskMoE`
- **Problem**: Package not on PyPI, not in `requirements.txt`, not in repo
- **Root cause**: Based on ModSquad (Chen et al. CVPR 2023). The exact package
  is a custom implementation distributed separately.
- **What it is**: The `parallel_linear/` subdirectory of the Mod-Squad repo
  (UMass Foundation Model), which installs itself as the `parallel_experts` package.
- **Fix applied**: ✅
  ```bash
  git clone https://github.com/UMass-Foundation-Model/Mod-Squad.git
  cd Mod-Squad/parallel_linear
  /opt/miniconda3/envs/EpiModX/bin/pip install --no-build-isolation .
  ```
  - `--no-build-isolation` required because setup.py imports torch at build time;
    pip's isolated build subprocess does not have access to the conda env's torch.
  - `numpy` had to be installed first in the EpiModX env before the above worked.
- **Classes used**:
  - `TaskMoE` — task-conditioned expert routing (disease group routing in attention
    Q-projection and FFN layers)
  - `RandomMoE` — random routing (ablation baseline)
- **Status**: ✅ RESOLVED — `parallel_experts 0.0.0` installed in EpiModX env

#### [C2] Missing benchmark model modules cause ImportError at startup
- **Files**: `train_MTL_Moe.py` lines 12–15, `test_MTL_Moe.py` lines 12–15, 24
- **Imports**: `sei`, `pretrain_multihead`, `DeepHistone`, `ablution_Study`
- **Problem**: These are baseline comparison models not included in the repo
- **Fix applied**: ✅ Commented out all four import lines in both scripts

#### [C3] Reference genome FASTA missing
- **File**: `utils/utils.py` line 18
- **Problem**: `faste_path = "Path to reference genome"` — literal placeholder string
- **Fix applied**: ✅ Changed to read from `REFERENCE_GENOME_PATH` environment variable
- **Reference file**: `reference/hg38.fa` — UCSC-style GRCh38 (chr1/chr2 naming, 3.0 GB)
  - FASTA index: `reference/hg38.fa.fai` (50 bp/line, matches hg38.fa)
  - Note: Must use UCSC-style FASTA (chr1/chr2 names), NOT NCBI-style (NC_000001.11)
    because BED files use chr* chromosome names.
- **Required action**: Set env var before any run:
  ```bash
  export REFERENCE_GENOME_PATH=/Users/shunjialou/Desktop/EpiModX/reference/hg38.fa
  # To make permanent:
  echo 'export REFERENCE_GENOME_PATH=/Users/shunjialou/Desktop/EpiModX/reference/hg38.fa' >> ~/.zshrc
  ```

#### [C4] Hardcoded absolute path in `mutationDataset`
- **File**: `utils/utils.py` line 75
- **Problem**: `/home/xiaoyu/Genome/data/human/genome/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta`
- **Fix applied**: ✅ Also reads from `REFERENCE_GENOME_PATH` environment variable

---

### BUGS — Runtime errors

#### [B1] `args.seed` undefined in `test_MTL_Moe.py`
- **File**: `test_MTL_Moe.py` line 71
- **Problem**: `torch.manual_seed(args.seed)` — `--seed` was never added to argparse
- **Fix applied**: ✅ Added `--seed` argument to argparse in test script

#### [B2] `-h` flag conflict in `test_MTL_Moe.py`
- **File**: `test_MTL_Moe.py` line 67
- **Problem**: `parser.add_argument("-h", "--histone", ...)` — `-h` is reserved by
  argparse for `--help`, causes a crash on initialization
- **Fix applied**: ✅ Changed short flag to `-t` (`-t`/`--histone`)

#### [B3] `model(data)` returns tuple in `test_MTL_Moe.py`
- **File**: `test_MTL_Moe.py` line 55
- **Problem**: `outputs = model(data).squeeze()` — `Pretrain_Moe.forward()` returns
  `(outputs_dict, aux_loss)` tuple, not a tensor. Calling `.squeeze()` on a tuple crashes.
- **Fix applied**: ✅ Unpacked to `outputs, _ = model(data)` and concatenated task outputs

---

### DATA PIPELINE — Missing files

#### [D1] `generate_dataset.py` missing
- **Problem**: The paper requires a specific data construction pipeline not shipped with the repo.
- **Pipeline spec** (from paper Methods):
  - Load ChIP-seq BED files per patient per histone
  - Merge overlapping peaks (>80% reciprocal overlap) across patients
  - Center each merged peak at midpoint, extend to 4096 bp
  - Define positives: ≥1 peak in central 2 kb OR >50% of sequence length is peak
  - Sample equal negatives from GRCh38, excluding ENCODE blacklist regions
  - Output per-histone CSV with 22 binary patient labels
- **Fix applied**: ✅ `generate_dataset.py` written from scratch per paper Methods
- **Performance bug fixed**: Original `is_positive()` used `continue` instead of `break`
  when iterating past the region end — caused O(all_peaks_per_chrom) per region instead
  of O(peaks_overlapping_region). Fixed to use `break` for the `pk_start >= region_end`
  condition.
- **Usage**:
  ```bash
  python generate_dataset.py --histone H3K27ac --blacklist reference/hg38-blacklist.v2.bed
  python generate_dataset.py --histone H3K4me3 --blacklist reference/hg38-blacklist.v2.bed
  python generate_dataset.py --histone H3K27me3 --blacklist reference/hg38-blacklist.v2.bed
  ```

#### [D2] Reference genome not downloaded
- **Directory**: `reference/`
- **Fix applied**: ✅ `reference/hg38.fa` downloaded (UCSC GRCh38, UCSC-style chr names)
- **Fix applied**: ✅ `reference/hg38.fa.fai` index placed (50 bp/line, matches hg38.fa)

#### [D3] ENCODE blacklist file not present
- **Problem**: Paper specifies excluding ENCODE blacklist regions for negative sampling
- **Fix applied**: ✅ `reference/hg38-blacklist.v2.bed` downloaded
  (source: github.com/Boyle-Lab/Blacklist)

#### [D4] Directory ambiguity (`Dataset/` vs `Datasets/`)
- **Problem**: `Dataset/` (no 's') contains `generated_all_data.csv` and `bigwig/`.
  `Datasets/` (with 's') contains `AD datasets.csv` and `bigwig/`. Two separate dirs.
- **Resolution**: Training scripts use `Datasets/{histone}_all_data.csv`. `generate_dataset.py`
  outputs to `Datasets/` to match. The `Dataset/` directory is from a previous experiment.

---

### ENVIRONMENT — Python dependencies

#### [E1] Most packages from `requirements.txt` not installed in EpiModX conda env
- **Env**: `/opt/miniconda3/envs/EpiModX/` (Python 3.10)
- **Problem**: After fresh conda env creation, only `torch` and `numpy` were present.
- **Fix applied**: ✅ All packages installed; numpy downgraded to 1.26.4 (numpy 2.x caused
  binary incompatibility with pandas 2.0.0 — `ValueError: numpy.dtype size changed`)
- **Note**: `requirements.txt` pins `numpy==2.2.6` but this is incompatible with pandas 2.0.0
  and torch 2.2.2 (both compiled against numpy 1.x). Actual installed: numpy 1.26.4.

#### [E2] `parallel_experts` build failure due to missing torch in isolated build env
- **Problem**: `pip install .` inside `parallel_linear/` failed with
  `ModuleNotFoundError: No module named 'torch'` because pip's build isolation
  subprocess does not inherit the conda environment.
- **Fix applied**: ✅ Used `--no-build-isolation` flag (see C1 above)

---

### HARDWARE LIMITATIONS — Apple Silicon (M1 Pro)

#### [H1] Full LLM model (Pretrain_Moe / LLM_Moe) cannot run on M1 Mac
- **Root cause**: Caduceus DNA language model backbone requires `mamba_ssm`, which uses
  custom CUDA kernels. There is no CPU or MPS fallback.
- **Error**: `ImportError: This modeling file requires mamba_ssm. Run pip install mamba_ssm`
  (even if installed, mamba_ssm requires CUDA at runtime)
- **Fix applied**: ✅ Switched `train_MTL_Moe.py` to use `model_name = "CNN_Moe"` for local
  training. The LLM variant requires a CUDA GPU (cloud or HPC cluster).

#### [H2] MPS (Apple GPU) incompatible with `parallel_experts` custom kernels
- **Problem**: `ParallelLinear` uses `@custom_fwd(cast_inputs=torch.float16)` + `torch.jit.script`
  with tensor splits that crash on MPS Metal with assertion errors:
  `[MPSNDArrayDescriptor sliceDimension:withSubrange:] error`
- **Fix applied**: ✅ Reverted device selection to `cuda if available else cpu` in both scripts.
  MPS was tried but is not viable for this model.

#### [H3] `parallel_experts` hardcodes `.cuda()` — crashes on non-CUDA machines
- **File**: `parallel_experts/moe.py` (installed package)
  - `TaskMoE.init_aux_statistics()`: `self.MI_task_gate = torch.zeros(...).cuda()`
  - `TaskMoE.update_aux_statistics()`: `MI_task_gate` device mismatch on forward pass
- **Fix applied**: ✅ Patched the installed package file directly:
  - `init_aux_statistics`: Changed `.cuda()` → `.to(device)` using existing parameters
  - `update_aux_statistics`: Added `self.MI_task_gate = self.MI_task_gate.to(probs.device)`
    before tensor update

#### [H4] Training speed on M1 Pro CPU
- **CNN_Moe on H3K4me3** (135K rows, batch=8):
  - ~13,500 steps/epoch × 20 epochs = 270K total steps
  - Evaluation every 30,000 steps (no console output before that)
  - **Estimated duration: 12–24 hours** on M1 Pro CPU
- **For faster iteration**: Reduce `global_step%30000 == 0` to e.g. `%1000 == 0`
  in `train_MTL_Moe.py` to see validation metrics sooner
- **For production (full LLM_Moe model)**: Requires CUDA GPU (cloud/HPC)

---

### DOWNSTREAM ANALYSIS — Not yet implemented

| Analysis | Paper section | Status |
|----------|--------------|--------|
| In silico mutagenesis (ISM) | Model interpretability | Not implemented |
| Gradient × Input scoring | Model interpretability | Captum installed, no script |
| Differential importance score (S_AD − S_nonAD) | Cross-disease analysis | Not implemented |
| haQTL prediction pipeline | Figure 4 | Not implemented |
| KEGG pathway enrichment | Figure 5 | External (WebGestalt) |
| S-LDSC heritability partitioning | Figure 5 | External tool |
| Motif enrichment (HOMER) | Figure 3 | External tool |

---

## Fix Summary

| ID | File | Action | Status |
|----|------|--------|--------|
| C1 | `Mod-Squad/parallel_linear/` | Installed `parallel_experts` with `--no-build-isolation` | ✅ Fixed |
| C2 | `train_MTL_Moe.py` | Commented out missing baseline imports | ✅ Fixed |
| C2 | `test_MTL_Moe.py` | Commented out missing baseline imports | ✅ Fixed |
| C3 | `utils/utils.py` | Reference genome path → `REFERENCE_GENOME_PATH` env var | ✅ Fixed |
| C4 | `utils/utils.py` | Hardcoded path → `REFERENCE_GENOME_PATH` env var | ✅ Fixed |
| B1 | `test_MTL_Moe.py` | Added `--seed` to argparse | ✅ Fixed |
| B2 | `test_MTL_Moe.py` | Changed `-h` to `-t` for `--histone` | ✅ Fixed |
| B3 | `test_MTL_Moe.py` | Fixed tuple unpacking from `model()` | ✅ Fixed |
| D1 | `generate_dataset.py` | Written from scratch per paper Methods | ✅ Created |
| D1 | `generate_dataset.py` | Fixed `break` vs `continue` O(n) → O(log n) per region | ✅ Fixed |
| D2 | `reference/hg38.fa` | Downloaded UCSC GRCh38 FASTA | ✅ Done |
| D2 | `reference/hg38.fa.fai` | Placed correct .fai index (50 bp/line) | ✅ Done |
| D3 | `reference/hg38-blacklist.v2.bed` | Downloaded ENCODE blacklist | ✅ Done |
| E1 | EpiModX conda env | Installed all requirements.txt packages; numpy→1.26.4 | ✅ Done |
| E2 | parallel_experts build | Used `--no-build-isolation` to bypass pip isolation | ✅ Done |
| H1 | `train_MTL_Moe.py` | Switched to CNN_Moe (LLM_Moe needs CUDA/mamba_ssm) | ✅ Worked around |
| H2 | `train_MTL_Moe.py`, `test_MTL_Moe.py` | Reverted MPS attempt; MPS crashes on ParallelLinear | ✅ Fixed |
| H3 | `parallel_experts/moe.py` (installed) | Patched hardcoded `.cuda()` → device-agnostic | ✅ Patched |

---

## How to Identify a Successful Training Run

Training prints to console **only at evaluation checkpoints** (every 30,000 global steps):
```
epochN
<validation_loss_float>
{'accuracy': tensor(...), 'AUC': tensor(...), 'F1': tensor(...), 'PRC': tensor(...)}
{'accuracy_test': ..., 'AUC_test': ..., 'F1_test': ..., 'PRC_test': ...}
```
**Success indicators**:
1. `models/H3K4me3_CNN_Moe.pt` file is created (only when `--save_model True` and validation loss improves)
2. Validation AUC > 0.6 (random = 0.5); paper reports AUC ~0.85–0.92 for the full LLM model
3. Loss decreases across checkpoints

**To see progress sooner**, change `train_MTL_Moe.py` line ~150:
```python
if global_step % 30000 == 0:   # original — very infrequent
# change to:
if global_step % 1000 == 0:    # see metrics every ~450 seconds on CPU
```

---

## Recommended Setup Steps

```bash
# 0. Activate EpiModX environment
conda activate EpiModX
# OR prefix all python/pip commands with /opt/miniconda3/envs/EpiModX/bin/

# 1. Set reference genome path (required before any run)
export REFERENCE_GENOME_PATH=/Users/shunjialou/Desktop/EpiModX/reference/hg38.fa
# Add to ~/.zshrc to make permanent

# 2. Generate training datasets — DONE (all 3 histones complete)
# H3K27ac: 434,110 rows  |  H3K4me3: 135,024 rows  |  H3K27me3: 623,356 rows

# 3. Train CNN_Moe on H3K4me3 (default; change histone in train_MTL_Moe.py line 32)
#    NOTE: LLM_Moe requires CUDA GPU — not available on M1 Mac
python train_MTL_Moe.py --save_model True
# Model saved to: models/H3K4me3_CNN_Moe.pt (when val loss improves)

# 4. Test
python test_MTL_Moe.py --histone H3K4me3
# Results saved to: test_results/H3K4me3_LLM_Moe_test_result (pickle)

# === For full LLM_Moe model (requires CUDA GPU): ===
# 1. Run on cloud (AWS/GCP/Colab) or HPC cluster with NVIDIA GPU
# 2. Install mamba_ssm: pip install mamba_ssm
# 3. Switch back: model_name = "LLM_Moe" in train_MTL_Moe.py
# 4. Use pretrain=True datasets (same CSVs work)
```

## Current Environment State

| Item | Path | State |
|------|------|-------|
| Conda env | `/opt/miniconda3/envs/EpiModX/` | Python 3.10, all deps installed |
| Reference FASTA | `reference/hg38.fa` | ✅ UCSC GRCh38, 3.0 GB |
| FASTA index | `reference/hg38.fa.fai` | ✅ 50 bp/line, matches hg38.fa |
| Blacklist | `reference/hg38-blacklist.v2.bed` | ✅ ENCODE hg38 blacklist v2 |
| BED files | `Datasets/bigwig/` | ✅ Per-patient ChIP-seq BEDs (22 × 3 histones) |
| H3K27ac CSV | `Datasets/H3K27ac_all_data.csv` | ✅ 434,110 rows |
| H3K4me3 CSV | `Datasets/H3K4me3_all_data.csv` | ✅ 135,024 rows |
| H3K27me3 CSV | `Datasets/H3K27me3_all_data.csv` | ✅ 623,356 rows |
| Trained model | `models/H3K4me3_CNN_Moe.pt` | 🔄 Training in progress (CPU, PID 63512) |
