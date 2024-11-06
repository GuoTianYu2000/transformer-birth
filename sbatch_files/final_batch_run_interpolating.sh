#!/bin/bash

conda init bash
source activate nanogpt

# Assuming the Python file is named 'script.py' and accepts parameters
# params=(0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)

# dormant copy task
layer_values=(1 3)
k_values=(3 5)
seed_list=(20 21)
mix_p_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
iter_num=10000

# Interpolating
for mix_p in "${mix_p_list[@]}"
do
for seed in "${seed_list[@]}"
do
  python ihead_full_main.py max_iters=${iter_num} log_probes=True eval_delta=5  task_name=dormant_copy_interpolate fine_grid_log=0 seperate_loss=True seed=53\
  model_args.n_layers=1 model_args.n_heads=1 model_args.dim=256\
  data_args.k=3 data_args.fixed_special_toks=True data_args.bos_num=1 data_args.mix_p=${mix_p}\
  wandb_args.name=dormant_copy_interpolate\
  optim_args.use_sgd=False optim_args.learning_rate=0.0003 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
  save_dir=/data/tianyu/birth/gens/final/dormant_copy_interpolate_seed${seed}
done
done
