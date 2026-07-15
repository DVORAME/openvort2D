#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort_avalanches
#SBATCH --output=output_avalanches/output_avalanches.out
#SBATCH --error=output_avalanches/error_avalanches.err

pyenv exec python src/main.py --N 600 --tmax 100 --dt 1e-5 --pin-type drag --D 0.1 --circle --omega-ex '2*600*KAPPA/np.pi/D**2*(1-2*600*KAPPA/np.pi/D**2*t/200/2/np.pi)' --pinning-v-ex 600*KAPPA/np.pi*0.9 \
	--polarization-type skewed --polarization 1 --save --save-every 1000 --no-plot --no-plot-save --output output_avalanches --gpu --load