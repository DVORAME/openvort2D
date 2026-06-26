#!/bin/bash
#SBATCH --partition=gpu-ffa
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=output_%j/log.txt
#SBATCH --error=output_%j/err.txt
#SBATCH --time=01:00:00
#SBATCH --mail-user=86205591@cuni.cz


alpha=0.061
alphap=0.01746
D=0.1
dt=1e-5
tmax=60

N=20000
v_pin=15
v_probe=25
freq=2000

python src/main.py --D $D --dt=$dt --alpha $alpha --alphap $alphap --walls --save --save-every 20 \
	--tmax $tmax --pinning-v $v_pin --probe-v $v_probe --probe-v-freq $freq --N $N --no-plot --gpu --output output_$SLURM_JOB_ID/out
