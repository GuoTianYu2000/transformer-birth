#!/bin/bash
#SBATCH -J massive       
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tianyu_guo@berkeley.edu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
conda init bash
source activate nanogpt

layer_values=(1 2 3)
head_values=(1)
layer_index=$(($SLURM_ARRAY_TASK_ID / ${#head_values[@]}))
head_index=$(($SLURM_ARRAY_TASK_ID % ${#head_values[@]}))
layer=${layer_values[$layer_index]}
head=${head_values[$head_index]}
python ihead_full_main.py max_iters=10000 log_probes=True eval_delta=5  model_args.n_layers=${layer} model_args.n_heads=${head} \
        data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1\
        data_args.delimiter_p=0 fine_grid_log=10000 wandb_args.name=dormant_dynamics seperate_loss=True\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 task_name=dormant_copy\
        model_args.dim=256 save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_copy/model_L${layer}_H${head}_bos1_delim0

# test code
# python ihead_full_main.py max_iters=5000 log_probes=True eval_delta=5  model_args.n_layers=1 model_args.n_heads=1 seperate_loss=True\
#         data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1\
#         data_args.delimiter_p=0 fine_grid_log=1000 wandb_args.name=dormant_test\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 task_name=dormant_copy\
#         model_args.dim=256 save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_test/model_L1_H1_bos1_delim0

# python ihead_full_main.py max_iters=1000 log_probes=True eval_delta=5  model_args.n_layers=1 model_args.n_heads=1 seperate_loss=True\
#         data_args.k=5 data_args.fixed_special_toks=True data_args.bos_num=1\
#         data_args.delimiter_p=0 fine_grid_log=500 wandb_args.name=markov\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 task_name=markov\
#         model_args.dim=256 save_dir=/data/tianyu_guo/birth/gens/pre_final/markov/model_L1_H1_bos1_delim0
