#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort_avalanches
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --array=1-1

set -euo pipefail

#
# Avalanche sweep parameters (override via environment variables).
#
D_BUFFER="${D}"
D=$(echo "scale=10; ${DD} * ${SLURM_ARRAY_TASK_ID} + ${D}" | bc -l)
N=$(echo "scale=0; $(echo "scale=10; ${N} * (${D} / ${D_BUFFER}) ^ 2" | bc -l)" / 1 | bc -l)
DT="${DT:-1e-5}"
TMAX="${TMAX:-33}"
PIN_TYPE="${PIN_TYPE:-drag}"
POLARIZATION_TYPE="${POLARIZATION_TYPE:-skewed}"
POLARIZATION="${POLARIZATION:-1}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
OMEGA_EXPRESSION=${OMEGA_EXPRESSION:-2*${N}*KAPPA/np.pi/D**2*(1-t/${TMAX})}
PINNING_V_EXPRESSION="${PINNING_V_EXPRESSION:-0}"
ALPHA="${ALPHA:-0.005*((${SLURM_ARRAY_TASK_ID}-1)//3)}"
ALPHA_PRIME="${ALPHA_PRIME:-0.005*((${SLURM_ARRAY_TASK_ID}-1)%3)}"
USE_GPU="${USE_GPU:-1}"

AVALANCHE_ROOT="${AVALANCHE_ROOT:-output_avalanches}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
TOTAL_TASKS="${TOTAL_TASKS:-1}"

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
	echo "ERROR: task id ${TASK_ID} is out of range 1..${TOTAL_TASKS}"
	echo "Set #SBATCH --array=1-${TOTAL_TASKS} or adjust TOTAL_TASKS."
	exit 1
fi


TASK_OUTPUT_DIR="${AVALANCHE_ROOT}/output_${TASK_ID}"
echo "[$(date)] Running avalanche task ${TASK_ID}/${TOTAL_TASKS}"

CMD=(
	pyenv exec python src/main.py
	--N "${N}"
	--tmax "${TMAX}"
	--dt "${DT}"
	--pin-type "${PIN_TYPE}"
	--D "${D}"
	--circle
	--omega-ex "${OMEGA_EXPRESSION}"
	--pinning-v-ex "${PINNING_V_EXPRESSION}"
	--polarization-type "${POLARIZATION_TYPE}"
	--polarization "${POLARIZATION}"
	--alpha "${ALPHA}"
	--alphap "${ALPHA_PRIME}"
	--save
	--save-every "${SAVE_EVERY}"
	--no-plot
	--no-plot-save
	--output "${TASK_OUTPUT_DIR}"
	--task-id "${TASK_ID}"
)

if [[ "${USE_GPU}" == "1" ]]; then
	CMD+=(--gpu)
fi

"${CMD[@]}"
