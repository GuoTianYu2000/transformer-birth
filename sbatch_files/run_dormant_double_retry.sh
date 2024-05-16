#!/bin/bash

conda init bash
source activate nanogpt

# Assuming the Python file is named 'script.py' and accepts parameters
# params=(0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)
params=(1 2)

# SLURM_ARRAY_TASK_ID=0

for n_heads in "${params[@]}"
do
  python ihead_full_main.py max_iters=10000 log_probes=True eval_delta=5  task_name=dormant_double_tasks_explore fine_grid_log=0 seperate_loss=True seed=27\
        model_args.n_layers=1 model_args.n_heads=${n_heads} model_args.dim=256\
        data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.delimiter_p=0 data_args.delim_num=1\
        wandb_args.name=dormant_double_tasks_explore\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_double_tasks_explore
done
