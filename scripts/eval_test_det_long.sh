GPUS=$1
CKPT=$2
PORT=$3
echo ${CKPT}
echo ${PORT}
export TORCH_DISTRIBUTED_DEBUG=INFO; python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=${PORT} \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root dataroot/ \
    --val_freq 5 --batch_size 32 --save_freq 5 --print_freq 50 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset es --test_dataset es \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ./logs/test_butd_long \
    --lr_decay_epochs 25 26 \
    --checkpoint_path=${CKPT} \
    --butd --self_attend --augment_det \
    --eval