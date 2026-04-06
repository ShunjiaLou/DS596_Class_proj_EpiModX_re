# Running EpiModX on BU SCC — Clone to Training to Plotting

Complete guide for reproducing the paper results on the Boston University Shared Computing Cluster.

---

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

Note the **full absolute path** to hg38.fa — you will need it in Steps 5 and 6.

---

## Step 3 — Create conda environment

```bash
module load miniconda
conda create -n EpiModX python=3.10 -y
conda activate EpiModX
pip install -r requirements.txt
```

---

## Step 4 — Install `mamba_ssm` (requires a GPU node)

`mamba_ssm` must be compiled against CUDA and cannot be installed on a login node.

```bash
qrsh -l gpus=1 -l gpu_c=6.0    # request interactive GPU session
conda activate EpiModX
pip install mamba_ssm
exit                             # return to login node when done
```

---

## Step 5 — Install `parallel_experts`

```bash
git clone https://github.com/UMass-Foundation-Model/Mod-Squad.git
cd Mod-Squad/parallel_linear
pip install --no-build-isolation .
cd ../..
```

---

## Step 6 — Edit the SLURM scripts

Two lines to update in **both** `sbatch_train.sh` and `sbatch_test.sh`:

```bash
nano sbatch_train.sh
nano sbatch_test.sh
```

| Line | Change |
|------|--------|
| `module load cuda/11.8` | Replace with the correct version (`module avail cuda`) |
| `export REFERENCE_GENOME_PATH=/path/to/hg38.fa` | Replace with actual path from Step 2 |

---

## Step 7 — Increase batch size for GPU

Edit `train_MTL_Moe.py` line ~36:

```bash
nano train_MTL_Moe.py
# Change: batch_size = 8
# To:     batch_size = 32   (or 64 if GPU has ≥ 40 GB VRAM)
```

---

## Step 8 — Submit training jobs (one per histone mark)

Each histone mark is trained as a separate SLURM job. Submit all three at once:

```bash
mkdir -p logs
for h in H3K27ac H3K4me3 H3K27me3; do
    sbatch --job-name=$h sbatch_train.sh $h
done
```

Monitor jobs:

```bash
qstat -u $USER                       # check job status
tail -f logs/train_H3K27ac_*.out     # watch live output for one job
```

> First validation metrics appear at step 30,000 (~3–4 hours per job on a single GPU).  
> Best model saved to `models/{HISTONE}_LLM_Moe.pt` when validation loss improves.

---

## Step 9 — Run evaluation after training

Once all three training jobs complete, submit the test jobs:

```bash
for h in H3K27ac H3K4me3 H3K27me3; do
    sbatch --job-name=$h sbatch_test.sh $h
done
```

Results are saved to:
- `test_results/H3K27ac_LLM_Moe_test_result`
- `test_results/H3K4me3_LLM_Moe_test_result`
- `test_results/H3K27me3_LLM_Moe_test_result`

---

## Step 10 — Generate figures

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
| 2 | Locate or download hg38.fa | ☐ |
| 3 | Create conda env + pip install | ☐ |
| 4 | Install mamba_ssm on GPU node | ☐ |
| 5 | Install parallel_experts | ☐ |
| 6 | Edit REFERENCE_GENOME_PATH + CUDA version in SLURM scripts | ☐ |
| 7 | Increase batch_size to 32 in train_MTL_Moe.py | ☐ |
| 8 | Submit 3 training jobs | ☐ |
| 9 | Submit 3 test jobs after training | ☐ |
| 10 | Run plot_results.py | ☐ |

---

## Things you need to figure out on SCC

| Item | How to find it |
|------|----------------|
| CUDA module name | `module avail cuda` |
