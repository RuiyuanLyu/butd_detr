GPUS=$1

export TORCH_DISTRIBUTED_DEBUG=INFO; python -m torch.distributed.launch --nproc_per_node ${GPUS} --master_port $RANDOM \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root dataroot/ \
    --val_freq 500 --batch_size 1 --save_freq 1000 --print_freq 1 --max_epoch 2000 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset es --test_dataset es \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/BUTD_addbutd_overfit \
    --lr_decay_epochs 900 1000 \
    --pp_checkpoint dataroot/gf_detector_l6o256.pth \
    --self_attend --debug