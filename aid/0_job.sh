#!/bin/bash -l

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%j.out

SCRIPT=""
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--script)
            SCRIPT="$2"
            shift 2
            ;;
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$SCRIPT" ]]; then
    echo "You must specify the script to run with --script <file.py>"
    exit 1
fi

unset SLURM_EXPORT_ENV

module load python
conda activate aid

srun python "$SCRIPT" "${SCRIPT_ARGS[@]}"
