#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort_prep_lattice
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

#
# Prep lattice parameters (override via environment variables).
#
N="${N:-600}"
D="${D:-0.1}"
DT="${DT:-1e-5}"
PREP_TMAX="${PREP_TMAX:-60}"
PIN_TYPE="${PIN_TYPE:-drag}"
POLARIZATION_TYPE="${POLARIZATION_TYPE:-skewed}"
POLARIZATION="${POLARIZATION:-1}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
PREP_OUTPUT_DIR="${PREP_OUTPUT_DIR:-output_prep_lattice}"
USE_GPU="${USE_GPU:-1}"

OMEGA_EXPRESSION="${OMEGA_EXPRESSION:-2*${N}*KAPPA/np.pi/D**2}"
PINNING_V_EXPRESSION="${PINNING_V_EXPRESSION:-${N}*KAPPA/np.pi*0.9}"

if [[ -e "${PREP_OUTPUT_DIR}" ]]; then
	echo "ERROR: '${PREP_OUTPUT_DIR}' already exists."
	echo "Use a new PREP_OUTPUT_DIR so the restart location is deterministic for run_avalanches.sh."
	exit 1
fi

echo "[$(date)] Preparing stable lattice in '${PREP_OUTPUT_DIR}'"
echo "N=${N}, D=${D}, dt=${DT}, prep_tmax=${PREP_TMAX}, save_every=${SAVE_EVERY}"

CMD=(
	pyenv exec python src/main.py
	--N "${N}"
	--tmax "${PREP_TMAX}"
	--dt "${DT}"
	--pin-type "${PIN_TYPE}"
	--D "${D}"
	--circle
	--omega-ex "${OMEGA_EXPRESSION}"
	--pinning-v-ex "${PINNING_V_EXPRESSION}"
	--polarization-type "${POLARIZATION_TYPE}"
	--polarization "${POLARIZATION}"
	--save
	--save-every "${SAVE_EVERY}"
	--no-plot
	--no-plot-save
	--output "${PREP_OUTPUT_DIR}"
	--alpha 0
	--alphap 0
)

if [[ "${USE_GPU}" == "1" ]]; then
	CMD+=(--gpu)
fi

"${CMD[@]}"

echo "[$(date)] Prep lattice done: ${PREP_OUTPUT_DIR}"
