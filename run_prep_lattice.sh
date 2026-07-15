#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort
#SBATCH --output=output_prep_lattice/output.out
#SBATCH --error=output_prep_lattice/error.err

echo "Running job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

pyenv exec python src/main.py --N 600 --tmax 10 --dt 1e-5 --pin-type drag --D 0.1 --circle --omega-ex '2*600*KAPPA/np.pi/D**2' --pinning-v-ex 600*KAPPA/np.pi*0.9 \
	--polarization-type skewed --polarization 1 --save --save-every 1000 --no-plot --no-plot-save --output output_prep_lattice --gpu