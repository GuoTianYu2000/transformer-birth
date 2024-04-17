#!/bin/bash
#SBATCH -J massive       
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tianyu_guo@berkeley.edu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=1-8
conda init bash
source activate nanogpt

layer_values=(1 2 3 4)
head_values=(2 4)
layer_index=$(($SLURM_ARRAY_TASK_ID / ${#head_values[@]}))
head_index=$(($SLURM_ARRAY_TASK_ID % ${#head_values[@]}))
layer=${layer_values[$layer_index]}
head=${head_values[$head_index]}
python ihead_full_main.py max_iters=5000 log_probes=True eval_delta=5  model_args.n_layers=${layer} model_args.n_heads=${head} \
        data_args.k=5 data_args.fixed_special_toks=True data_args.bos_num=1\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        model_args.dim=256 save_dir=/data/tianyu_guo/birth/gens/model_L${layer}_H${head}_lr3-4_bos1

# python ihead_full_main.py max_iters=5000 log_probes=True eval_delta=5  model_args.n_layers=3 model_args.n_heads=2 \
#         data_args.k=5 data_args.fixed_special_toks=True data_args.bos_num=0\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
#         model_args.dim=256 save_dir=/data/tianyu_guo/birth/gens/model_L3_H2_lr3-4

