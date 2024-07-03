GPUS=$1
CKPT=$2

export TORCH_DISTRIBUTED_DEBUG=INFO; python -m torch.distributed.launch --nproc_per_node ${GPUS} --master_port $RANDOM \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root dataroot/ \
    --val_freq 5 --batch_size 36 --save_freq 5 --print_freq 20 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset es --test_dataset es \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/BUTD_addbutd_resume \
    --lr_decay_epochs 25 26 \
    --checkpoint_path=${CKPT} \
    --pp_checkpoint dataroot/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det