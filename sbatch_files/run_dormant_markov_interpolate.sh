#!/bin/bash
#SBATCH -J massive       
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tianyu_guo@berkeley.edu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-10
conda init bash
source activate nanogpt

mix_p_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
mix_p=${mix_p_list[$SLURM_ARRAY_TASK_ID]}
python ihead_full_main.py max_iters=10000 log_probes=True eval_delta=5  task_name=dormant_markov_interpolate fine_grid_log=0 seperate_loss=True\
        model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
        data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=${mix_p}\
        wandb_args.name=dormant_markov_interpolate\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_markov_interpolate/model_L1_H1_bos1_delim0_${SLURM_ARRAY_TASK_ID}

# python ihead_full_main.py max_iters=100 log_probes=True eval_delta=5  task_name=dormant_markov_interpolate fine_grid_log=0 seperate_loss=True\
#         model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
#         data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=${mix_p}\
#         wandb_args.name=dormant_markov_interpolate_test\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
#         save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_markov_interpolate/model_L1_H1_bos1_delim0_${SLURM_ARRAY_TASK_ID}

# python ihead_full_main.py max_iters=1000 log_probes=True eval_delta=5  task_name=dormant_markov_interpolate fine_grid_log=0 seperate_loss=True\
#         model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
#         data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=0.3\
#         wandb_args.name=dormant_markov_interpolate_test\
#         optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
#         save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_markov_interpolate/model_L1_H1_bos1_delim0_3
