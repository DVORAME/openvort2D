#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort_avalanches
#SBATCH --array=0-7
#SBATCH --output=output_avalanches_1/output_avalanches_%a.out
#SBATCH --error=output_avalanches_1/error_avalanches_%a.err

echo "Running job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

pyenv exec python src/main.py --N 10000 --tmax 100 --dt 1e-4 --pin-type drag --D 1 --circle --omega-lambda '20000*KAPPA/np.pi-0.005*t' --pinning-v-func 10000*KAPPA/np.pi*\(1-0.01*$SLURM_ARRAY_TASK_ID\) \
	--polarization-type skewed --polarization 1 --save --save-every 1000 --no-plot --no-plot-save --output output_avalanches_1/output_$SLURM_ARRAY_TASK_ID --gpu