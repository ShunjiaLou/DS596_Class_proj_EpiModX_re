#!/bin/bash
#SBATCH -J EpiModX_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/test_%x_%j.out
#SBATCH -e logs/test_%x_%j.err

# Usage:
#   sbatch --job-name=H3K27ac  sbatch_test.sh H3K27ac
#   sbatch --job-name=H3K4me3  sbatch_test.sh H3K4me3
#   sbatch --job-name=H3K27me3 sbatch_test.sh H3K27me3
#
# Or run all three after training:
#   for h in H3K27ac H3K4me3 H3K27me3; do sbatch --job-name=$h sbatch_test.sh $h; done

HISTONE=${1:-H3K4me3}

# ── Load modules ───────────────────────────────────────────────────────────────
module load miniconda
module load cuda/11.8   # match the version used for training

# ── Activate environment ───────────────────────────────────────────────────────
conda activate EpiModX

# ── Reference genome path ──────────────────────────────────────────────────────
export REFERENCE_GENOME_PATH=/path/to/hg38.fa   # <-- EDIT THIS

# ── Run test ───────────────────────────────────────────────────────────────────
echo "Testing histone: $HISTONE"
python test_MTL_Moe.py --histone $HISTONE

echo "Testing finished. Results in test_results/${HISTONE}_LLM_Moe_test_result"
