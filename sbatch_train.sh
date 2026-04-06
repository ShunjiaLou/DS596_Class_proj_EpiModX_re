#!/bin/bash
#SBATCH -J EpiModX_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/train_%x_%j.out
#SBATCH -e logs/train_%x_%j.err

# Usage:
#   sbatch --job-name=H3K27ac  sbatch_train.sh H3K27ac
#   sbatch --job-name=H3K4me3  sbatch_train.sh H3K4me3
#   sbatch --job-name=H3K27me3 sbatch_train.sh H3K27me3
#
# Or submit all three at once:
#   for h in H3K27ac H3K4me3 H3K27me3; do sbatch --job-name=$h sbatch_train.sh $h; done

HISTONE=${1:-H3K4me3}    # default to H3K4me3 if no argument given

# ── Load modules (adjust to your SCC module names) ────────────────────────────
module load miniconda
module load cuda/11.8   # check available versions: module avail cuda

# ── Activate environment ───────────────────────────────────────────────────────
conda activate EpiModX

# ── Reference genome path ──────────────────────────────────────────────────────
export REFERENCE_GENOME_PATH=/path/to/hg38.fa   # <-- EDIT THIS before submitting

# ── Run training ───────────────────────────────────────────────────────────────
echo "Training histone: $HISTONE"
python train_MTL_Moe.py --histone $HISTONE --save_model True

echo "Training finished: $HISTONE"
