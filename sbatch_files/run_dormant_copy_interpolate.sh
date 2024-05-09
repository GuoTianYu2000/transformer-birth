#!/bin/bash
#SBATCH -J massive       
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tianyu_guo@berkeley.edu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
conda init bash
source activate nanogpt

layer_values=(1)
head_values=(2)
layer_index=$(($SLURM_ARRAY_TASK_ID / ${#head_values[@]}))
head_index=$(($SLURM_ARRAY_TASK_ID % ${#head_values[@]}))
layer=${layer_values[$layer_index]}
head=${head_values[$head_index]}
python ihead_full_main.py max_iters=10000 log_probes=True eval_delta=5  task_name=dormant_copy_interpolate fine_grid_log=0 seperate_loss=True\
        model_args.n_layers=${layer} model_args.n_heads=${head} model_args.dim=256\
        data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=0.1\
        wandb_args.name=dormant_copy_interpolate\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_copy_interpolate/model_L${layer}_H${head}_bos1_delim005

# python ihead_full_main.py max_iters=1000 log_probes=True eval_delta=5  task_name=dormant_copy_interpolate fine_grid_log=0 seperate_loss=True\
#         model_args.n_layers=1 model_args.n_heads=2 model_args.dim=256\
#         data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=0.1\
#         wandb_args.name=dormant_copy_interpolate_test\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
#         save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_copy_interpolate/model_L1_H2_bos1_delim005

