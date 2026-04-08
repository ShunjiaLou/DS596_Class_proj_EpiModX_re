# Running EpiModX on BU SCC — Clone to Training to Plotting

Complete guide for reproducing the paper results on the Boston University Shared Computing Cluster.

---

# Part 1: Running EpiModX (Step-by-Step)

## Step 1 — Clone the repo

```bash
cd /projectnb/YOURLAB    # replace with your lab/project directory
git clone https://github.com/ShunjiaLou/DS596_Class_proj_EpiModX_re.git
cd DS596_Class_proj_EpiModX_re
```

---

## Step 2 — Download hg38.fa

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
module load samtools
samtools faidx hg38.fa
```

Note the **full absolute path** to hg38.fa — it is already set in the job scripts (`sbatch_train.sh` / `sbatch_test.sh`).

---

## Step 3 — Create conda environment (login node)

```bash
module load miniconda
conda create -n EpiModX_cuda122 python=3.10 -y
source /share/pkg.8/miniconda/25.3.1/install/etc/profile.d/conda.sh
conda activate EpiModX_cuda122
```

---

## Step 4 — Install all packages (GPU node)

PyTorch and `mamba_ssm` require CUDA and must be installed on an interactive GPU session.

Request a GPU node:

```bash
qrsh -P ds596 -l gpus=1 -l gpu_c=7.0
```

Once on the GPU node, load modules and activate the environment:

```bash
module load miniconda
module load cuda/12.2
module load gcc/12.2.0

source /share/pkg.8/miniconda/25.3.1/install/etc/profile.d/conda.sh
conda activate EpiModX_cuda122

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Install all requirements (includes PyTorch with CUDA index):

```bash
pip install -r requirements.txt
pip install "numpy<2"    # avoid NumPy 2.x binary incompatibility
```

Install `mamba_ssm`:

```bash
pip install --no-build-isolation mamba-ssm
```

Install `parallel_experts`:

```bash
git clone https://github.com/UMass-Foundation-Model/Mod-Squad.git
cd Mod-Squad/parallel_linear
pip install --no-build-isolation .
cd ../..
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Exit the GPU session when done:

```bash
exit
```

---

## Step 5 — Increase batch size for GPU

Edit `train_MTL_Moe.py` line ~36:

```bash
nano train_MTL_Moe.py
# Change: batch_size = 8
# To:     batch_size = 32   (or 64 if GPU has ≥ 40 GB VRAM)
```

---

## Step 6 — Submit training jobs (one per histone mark)

Each histone mark is trained as a separate SGE job. Submit all three at once:

```bash
mkdir -p logs
for h in H3K27ac H3K4me3 H3K27me3; do
    qsub -N "$h" sbatch_train.sh "$h"
done
```

Monitor jobs:

```bash
qstat -u $USER                           # check job status (qw=waiting, r=running)
tail -f logs/train_H3K27ac_*.out         # watch live output for one job
```

> First validation metrics appear at step 30,000 (~3–4 hours per job on a single GPU).  
> Best model saved to `models/{HISTONE}_LLM_Moe.pt` when validation loss improves.

---

## Step 7 — Run evaluation after training

Once all three training jobs complete, submit the test jobs:

```bash
for h in H3K27ac H3K4me3 H3K27me3; do
    qsub -N "$h" sbatch_test.sh "$h"
done
```

Results are saved to:
- `test_results/H3K27ac_LLM_Moe_test_result`
- `test_results/H3K4me3_LLM_Moe_test_result`
- `test_results/H3K27me3_LLM_Moe_test_result`

---

## Step 8 — Generate figures

After all three test jobs complete:

```bash
python plot_results.py
```

Outputs:
- `figures/fig2_per_patient_metrics.png` — per-patient AUROC, AUPRC, F1, ACC (Figure 2a-c)
- `figures/fig3ab_cross_disease_precision_recall.png` — cross-disease precision/recall bubble plots (Figure 3a-b)

---

## Quick-reference checklist

| # | Task | Done? |
|---|------|-------|
| 1 | Clone repo (datasets included) | ☐ |
| 2 | Download hg38.fa | ☐ |
| 3 | Create conda env (login node) | ☐ |
| 4 | Install all packages on GPU node (PyTorch, mamba_ssm, parallel_experts) | ☐ |
| 5 | Increase batch_size to 32 in train_MTL_Moe.py | ☐ |
| 6 | Submit 3 training jobs | ☐ |
| 7 | Submit 3 test jobs after training | ☐ |
| 8 | Run plot_results.py | ☐ |

---

# Part 2: GPU Environment — Troubleshooting & Reference

## Accessing a GPU Node

```bash
qrsh -P ds596 -l gpus=1              # standard request
qrsh -P ds596 -l gpus=1 -l gpu_c=7.0  # request newer GPU (V100+)
```

## One-Line Activation (after first-time setup)

```bash
module load cuda/12.2 gcc/12.2.0 miniconda && source activate && conda activate EpiModX_cuda122
```

## Quick Debug

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `pip not found` | env not activated | `source activate && conda activate EpiModX_cuda122` |
| `nvcc not found` | forgot module load | `module load cuda/12.2` |
| CUDA mismatch | torch CUDA != system CUDA | reinstall torch with `--extra-index-url https://download.pytorch.org/whl/cu121` |
| GCC too old | old compiler | `module load gcc/12.2.0` |
| NumPy crash | NumPy 2.x | `pip install "numpy<2"` |
| `sbatch not found` | wrong scheduler | use `qsub` (BU SCC uses SGE, not SLURM) |

## Key Mental Model

You must align 3 layers:

1. **GPU node** — `qrsh`
2. **System modules** — CUDA 12.2 + GCC 12.2
3. **Python env** — torch 2.2.2 + packages

If any layer is mismatched → things break.
