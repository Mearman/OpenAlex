#!/bin/bash
#SBATCH --job-name=sync-works
#SBATCH --partition=htc_genoa
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/SCWF00070/b.abs217/logs/sync-works-%j.log

# OpenAlex works entity extraction on Falcon HPC.
# Idempotent: skips parquets that already exist (provenance tracking).
# Auto-sizes workers from job memory limit (cgroup/Slurm env).

set -euo pipefail

source /scratch/SCWF00070/b.abs217/.venv/bin/activate
cd /scratch/SCWF00070/b.abs217/OpenAlex
export OPENALEX_SYNC_ROOT=/scratch/SCWF00070/b.abs217/openalex-snapshot
export LOG_DIR=/scratch/SCWF00070/b.abs217/logs

mkdir -p "$LOG_DIR"

echo "=== sync-works on $(hostname) at $(date) ==="
echo "Job: $SLURM_JOB_ID, Partition: $SLURM_JOB_PARTITION, CPUs: $SLURM_CPUS_PER_TASK, Mem: ${SLURM_MEM_PER_NODE:-?}M"

python3 -m sync --no-upload --entity works

echo "=== finished at $(date) ==="
