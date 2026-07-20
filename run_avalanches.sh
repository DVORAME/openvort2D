#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort_avalanches
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --array=1-5

set -euo pipefail

#
# Avalanche sweep parameters (override via environment variables).
#
N="${N:-600}"
D="${D:-0.1}"
DT="${DT:-1e-5}"
SPINDOWN_TMAX="${SPINDOWN_TMAX:-33}"
SPINDOWN_BASE_TIME="${SPINDOWN_BASE_TIME:-50}"
PIN_TYPE="${PIN_TYPE:-drag}"
POLARIZATION_TYPE="${POLARIZATION_TYPE:-skewed}"
POLARIZATION="${POLARIZATION:-1}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
USE_GPU="${USE_GPU:-1}"

PREP_OUTPUT_DIR="${PREP_OUTPUT_DIR:-output_prep_lattice}"
AVALANCHE_ROOT="${AVALANCHE_ROOT:-output_avalanches}"

# One array item uses one factor. With default #SBATCH --array=1-5, keep 5 values here.
SPINDOWN_RATE_FACTORS=(${SPINDOWN_RATE_FACTORS:-1 2 3 4 5})

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
TOTAL_TASKS="${#SPINDOWN_RATE_FACTORS[@]}"

if (( TASK_ID < 1 || TASK_ID > TOTAL_TASKS )); then
	echo "ERROR: task id ${TASK_ID} is out of range 1..${TOTAL_TASKS}"
	echo "Set #SBATCH --array=1-${TOTAL_TASKS} or adjust SPINDOWN_RATE_FACTORS."
	exit 1
fi

if [[ ! -d "${PREP_OUTPUT_DIR}" ]]; then
	echo "ERROR: prep output directory '${PREP_OUTPUT_DIR}' not found."
	echo "Run run_prep_lattice.sh first or point PREP_OUTPUT_DIR to that run output."
	exit 1
fi

mapfile -t PREP_RESTART_FILES < <(find "${PREP_OUTPUT_DIR}" -maxdepth 1 -name "*.npz" -type f | sort)
if [[ "${#PREP_RESTART_FILES[@]}" -eq 0 ]]; then
	echo "ERROR: no restart .npz file found in '${PREP_OUTPUT_DIR}'."
	exit 1
fi
last_restart_index=$((${#PREP_RESTART_FILES[@]} - 1))
SOURCE_RESTART_FILE="${PREP_RESTART_FILES[${last_restart_index}]}"
if restart_mtime=$(stat -c %Y "${SOURCE_RESTART_FILE}" 2>/dev/null); then
	:
else
	restart_mtime=$(stat -f %m "${SOURCE_RESTART_FILE}")
fi
SETUP_SIGNATURE="${SOURCE_RESTART_FILE}|${restart_mtime}|${TOTAL_TASKS}"

SETUP_DONE_FILE="${AVALANCHE_ROOT}/.setup_done"
SETUP_LOCK_DIR="${AVALANCHE_ROOT}/.setup_lock"
TARGET_RESTART_NAME="vp_00000000.npz"

mkdir -p "${AVALANCHE_ROOT}"
if mkdir "${SETUP_LOCK_DIR}" 2>/dev/null; then
	rm -f "${SETUP_DONE_FILE}"
	for ((i = 1; i <= TOTAL_TASKS; i++)); do
		run_dir="${AVALANCHE_ROOT}/output_${i}"
		mkdir -p "${run_dir}"
		cp -f "${SOURCE_RESTART_FILE}" "${run_dir}/${TARGET_RESTART_NAME}"
	done
	echo "${SETUP_SIGNATURE}" > "${SETUP_DONE_FILE}"
	rmdir "${SETUP_LOCK_DIR}"
else
	max_wait_s=300
	wait_step_s=2
	waited_s=0
	while true; do
		if [[ -f "${SETUP_DONE_FILE}" ]] && [[ "$(cat "${SETUP_DONE_FILE}")" == "${SETUP_SIGNATURE}" ]]; then
			break
		fi
		if (( waited_s >= max_wait_s )); then
			echo "ERROR: timed out waiting for setup to complete."
			exit 1
		fi
		sleep "${wait_step_s}"
		waited_s=$((waited_s + wait_step_s))
	done
fi

RATE_FACTOR="${SPINDOWN_RATE_FACTORS[$((TASK_ID - 1))]}"
TASK_OUTPUT_DIR="${AVALANCHE_ROOT}/output_${TASK_ID}"

OMEGA_EXPRESSION="2*${N}*KAPPA/np.pi/D**2*(1-2*${N}*KAPPA/np.pi/D**2*t/${SPINDOWN_BASE_TIME}/${RATE_FACTOR}/2/np.pi)"
PINNING_V_EXPRESSION="${PINNING_V_EXPRESSION:-${N}*KAPPA/D/np.pi/500*3}"

echo "[$(date)] Running avalanche task ${TASK_ID}/${TOTAL_TASKS}"
echo "prep='${PREP_OUTPUT_DIR}', restart='${SOURCE_RESTART_FILE}', output='${TASK_OUTPUT_DIR}', rate_factor=${RATE_FACTOR}"

CMD=(
	pyenv exec python src/main.py
	--N "${N}"
	--tmax "${SPINDOWN_TMAX}"
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
	--output "${TASK_OUTPUT_DIR}"
	--load
)

if [[ "${USE_GPU}" == "1" ]]; then
	CMD+=(--gpu)
fi

"${CMD[@]}"
