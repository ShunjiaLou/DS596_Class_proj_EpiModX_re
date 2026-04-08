#!/bin/bash
#$ -N EpiModX_test
#$ -P ds596
#$ -l gpus=1
#$ -l h_rt=02:00:00
#$ -cwd
#$ -o logs/test_$JOB_NAME_$JOB_ID.out
#$ -e logs/test_$JOB_NAME_$JOB_ID.err

HISTONE=${1:-H3K4me3}

module load miniconda
module load cuda/12.2
module load gcc/12.2.0

source /share/pkg.8/miniconda/25.3.1/install/etc/profile.d/conda.sh
conda activate EpiModX_cuda122

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export REFERENCE_GENOME_PATH="/projectnb/ds596/projects/Team 5/reference/hg38.fa"

echo "Testing histone: $HISTONE"
python test_MTL_Moe.py --histone "$HISTONE"

echo "Testing finished: $HISTONE"
echo "Results in test_results/${HISTONE}_LLM_Moe_test_result"
