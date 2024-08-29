#!/bin/bash

conda init bash
source activate nanogpt

# Assuming the Python file is named 'script.py' and accepts parameters
params=(0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18)

for mix_p in "${params[@]}"
do
  python ihead_full_main.py max_iters=10000 log_probes=True eval_delta=5  task_name=dormant_markov_interpolate fine_grid_log=0 seperate_loss=True\
        model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
        data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=${mix_p}\
        wandb_args.name=dormant_markov_interpolate\
        optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        save_dir=/data/tianyu_guo/birth/gens/pre_final/dormant_markov_interpolate
done
