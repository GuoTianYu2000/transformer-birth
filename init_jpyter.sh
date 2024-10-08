#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

source /data/[username]/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate nanogpt
jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.kernel_name=nanogpt
