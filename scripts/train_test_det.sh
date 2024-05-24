TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root dataroot/ \
    --val_freq 5 --batch_size 24 --save_freq 50 --print_freq 300 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset es --test_dataset es \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/bdetr \
    --lr_decay_epochs 25 26 \
    --pp_checkpoint dataroot/gf_detector_l6o256.pth \
    --self_attend --augment_det
