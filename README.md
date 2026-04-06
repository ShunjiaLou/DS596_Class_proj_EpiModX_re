# Reproducing EpiModX — DS596 Class Project

> **Course**: DS596 — Boston University  
> **Original paper**: Wang et al., *"Predicting disease-specific histone modifications and functional effects of non-coding variants by leveraging DNA language models"*, Genome Biology (2026)  
> **Original repository**: [yanwu20/EpiModX](https://github.com/yanwu20/EpiModX)  
> **This repository**: Reproduction effort, bug fixes, and extended setup for BU SCC cluster

---

## Overview

This project reproduces the EpiModX model from the above paper as part of DS596. EpiModX is a multi-task learning framework for predicting disease-specific histone modifications in Alzheimer's disease (AD) using a DNA language model (Caduceus) combined with a Mixture-of-Experts (MoE) architecture.

The original codebase had several blockers that prevented it from running. This repository documents and fixes all of them, implements the missing data pipeline, and provides scripts for training on the BU SCC GPU cluster.

---

## What This Repository Adds

| Contribution | Description |
|-------------|-------------|
| `GAP_ANALYSIS.md` | Full audit of the original code — every bug, missing file, and fix |
| `generate_dataset.py` | Data pipeline written from scratch per paper Methods section |
| `Datasets/bigwig/` | Per-patient ChIP-seq BED files (H3K27ac, H3K4me3, H3K27me3) |
| `Datasets/*_all_data.csv` | Generated training CSVs (434K / 135K / 623K rows per histone) |
| `reference/hg38-blacklist.v2.bed` | ENCODE GRCh38 blacklist for negative sampling |
| `sbatch_train.sh` | SLURM job script for SCC GPU training |
| `sbatch_test.sh` | SLURM job script for SCC evaluation |
| Bug fixes | 7 bugs fixed in `train_MTL_Moe.py`, `test_MTL_Moe.py`, `utils/utils.py` |

### Bugs fixed in original code
- **C2**: Missing baseline model imports cause `ImportError` at startup — commented out
- **C3/C4**: Hardcoded FASTA paths (`/home/xiaoyu/...`) replaced with `REFERENCE_GENOME_PATH` env var
- **B1**: `args.seed` referenced but never defined in `test_MTL_Moe.py`
- **B2**: `-h` flag conflicts with argparse `--help` in `test_MTL_Moe.py`
- **B3**: `model()` returns `(outputs, aux_loss)` tuple — `.squeeze()` call on tuple fixed
- **E1**: `parallel_experts` (Mod-Squad) not on PyPI — install instructions and `.cuda()` patch for non-CUDA hardware

---

## Setup

### Requirements
- Python 3.10
- CUDA GPU (for full LLM_Moe model — required for `mamba_ssm`)
- `hg38.fa` reference genome (not included — 3 GB)

### Install

```bash
# Clone
git clone https://github.com/ShunjiaLou/DS596_Class_Proj_EpiModD.git
cd DS596_Class_Proj_EpiModD

# Create environment
conda create -n EpiModX python=3.10 -y
conda activate EpiModX
pip install -r requirements.txt
pip install mamba_ssm          # requires CUDA GPU

# Install parallel_experts (MoE routing — not on PyPI)
git clone https://github.com/UMass-Foundation-Model/Mod-Squad.git
cd Mod-Squad/parallel_linear
pip install --no-build-isolation .
cd ../..
```

### Reference genome

```bash
# Download GRCh38 (UCSC-style, chr1/chr2 names — required for BED compatibility)
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
samtools faidx hg38.fa
export REFERENCE_GENOME_PATH=/path/to/hg38.fa
```

---

## Usage

### Generate training datasets (already included in repo)

```bash
export REFERENCE_GENOME_PATH=/path/to/hg38.fa
python generate_dataset.py --histone H3K27ac --blacklist reference/hg38-blacklist.v2.bed
python generate_dataset.py --histone H3K4me3 --blacklist reference/hg38-blacklist.v2.bed
python generate_dataset.py --histone H3K27me3 --blacklist reference/hg38-blacklist.v2.bed
```

### Train on SCC (GPU required)

```bash
# Edit REFERENCE_GENOME_PATH in sbatch_train.sh first
sbatch sbatch_train.sh
```

### Train locally (CPU only — CNN_Moe variant, no LLM)

```bash
# In train_MTL_Moe.py: set model_name = "CNN_Moe"
export REFERENCE_GENOME_PATH=/path/to/hg38.fa
python train_MTL_Moe.py --save_model True
```

### Test

```bash
sbatch sbatch_test.sh
# or locally:
python test_MTL_Moe.py --histone H3K4me3
```

---

## Original Paper Abstract

> Epigenetic modifications play a vital role in the pathogenesis of human diseases, particularly neurodegenerative disorders such as Alzheimer's disease (AD). We developed a novel LLM-based deep learning framework for disease-contextual prediction of histone modifications and variant effects. A key innovation is the incorporation of a Mixture of Experts (MoE) architecture, achieving mean AUROCs ranging from 0.7863 to 0.9142, significantly outperforming existing state-of-the-art methods.

![EpiModX Framework](https://github.com/user-attachments/assets/a97d4f5f-06dc-4d20-bdb9-d93c1dc19bdc)

*Figure from Wang et al. (2026)*

---

## Citation

If you use the original EpiModX model, please cite:

```
Wang et al., "Predicting disease-specific histone modifications and functional
effects of non-coding variants by leveraging DNA language models",
Genome Biology, 2026.
```

Original code: https://github.com/yanwu20/EpiModX  
Original contact: xiaoyu.wang2@monash.edu

---

## Repository Structure

```
├── train_MTL_Moe.py        # Training script (modified from original)
├── test_MTL_Moe.py         # Evaluation script (modified from original)
├── Pretrain_Moe.py         # Model architecture (original)
├── requirements.txt        # Python dependencies (original)
├── generate_dataset.py     # Data pipeline (added — not in original repo)
├── sbatch_train.sh         # SCC SLURM training job (added)
├── sbatch_test.sh          # SCC SLURM test job (added)
├── GAP_ANALYSIS.md         # Full reproduction audit log (added)
├── utils/
│   ├── utils.py            # Data loading utilities (modified from original)
│   └── Moe.py              # MoE module (original)
├── Datasets/
│   ├── AD datasets.csv     # Patient metadata
│   ├── bigwig/             # Per-patient ChIP-seq BED files
│   ├── H3K27ac_all_data.csv
│   ├── H3K4me3_all_data.csv
│   └── H3K27me3_all_data.csv
└── reference/
    └── hg38-blacklist.v2.bed
```
