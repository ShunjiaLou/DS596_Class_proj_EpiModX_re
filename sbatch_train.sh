#!/bin/bash
#$ -N EpiModX_train
#$ -P ds596
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l mem_per_core=32G
#$ -cwd
#$ -o logs/train_$JOB_NAME_$JOB_ID.out
#$ -e logs/train_$JOB_NAME_$JOB_ID.err

HISTONE=${1:-H3K4me3}

module load miniconda
module load cuda/12.2
module load gcc/12.2.0

source activate
conda activate EpiModX

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export REFERENCE_GENOME_PATH="/projectnb/ds596/projects/Team 5/reference/hg38.fa"

echo "Training histone: $HISTONE"
python train_MTL_Moe.py --histone "$HISTONE" --save_model True
echo "Training finished: $HISTONE"
