#!/bin/bash

export EXPR_ID=base-5-group
export DATA_DIR=/mnt/d/research/datasets
export CHECKPOINT_DIR=/mnt/d/research/nvae_models
export CODE_DIR=/home/ihowell/Projects/validity/validity/generators/nvae

python -m validity.generators.nvae.train --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 32 --num_channels_dec 32 --epochs 200 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 5 --batch_size 32 \
        --weight_decay_norm 1e-2 --num_nf 1 --num_process_per_node 1 --use_se --res_dist --fast_adamax
        # --mutation_rate 0.1
