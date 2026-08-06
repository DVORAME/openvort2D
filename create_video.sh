#!/bin/bash
#SBATCH -p gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=openvort
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

INPUT=${INPUT:-'output'}
OUTPUT_DIR=${OUTPUT_DIR:-'videos'}
FILE_NAME=${FILE_NAME:-"openvort_${SLURM_JOB_ID}.mp4"}
FRAMERATE=${FRAMERATE:-30}
INFO=${INFO:-1}
DPI=${DPI:-300}

echo "[$(date)] Creating video from '${INPUT}' in '${OUTPUT_DIR}'"
CMD_FRAMES=(
	pyenv exec python src/imager.py
	--input "${INPUT}"
	--output "/scratch/tmp/openvort_video_${SLURM_JOB_ID}"
	--save
	--dpi "${DPI}"
)

if [[ "${INFO}" == "1" ]]; then
	CMD_FRAMES+=(--info)
fi

echo "[$(date)] Creating frames in '/scratch/tmp/openvort_video_${SLURM_JOB_ID}'"

"${CMD_FRAMES[@]}"

CMD_VIDEO=(
	ffmpeg -framerate "${FRAMERATE}" -i "/scratch/tmp/openvort_video_${SLURM_JOB_ID}/frame_%08d.png" \
	-c:v h264 -pix_fmt yuv420p "${OUTPUT_DIR}/${FILE_NAME}"
)

echo "[$(date)] Creating video '${OUTPUT_DIR}/${FILE_NAME}' from frames in '/scratch/tmp/openvort_video_${SLURM_JOB_ID}'"

"${CMD_VIDEO[@]}"

rm -rf "/scratch/tmp/openvort_video_${SLURM_JOB_ID}"

echo "[$(date)] Video created: '${OUTPUT_DIR}/${FILE_NAME}'"
