#!/bin/bash

echo "Select which mode to run:"
echo "1) Baseline"
echo "2) Causal"
echo "3) Conf"
echo "4) CaConf"


read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        echo "Running Baseline..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        MOSCARD/config/Mayo.json \
        MCAT_vit_losssave \
        --num_workers 8 --device_ids "0" \
        --pre_train_backbone \
        /media/Datacenter_storage/jialu/ECG/logdir/logdir-pretrained_vit/best2.ckpt \
        /media/Datacenter_storage/jialu_/jialu_causalv2/logdir/logdir_pretrain_vit_new/best3.ckpt \
        --model "main" --logtofile True
        ;;
    2)
        echo "Running Causal..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json \
        MCAT_vit_losssave_causal \
        --num_workers 8 --device_ids "3" \
        --pre_train_backbone \
        /media/Datacenter_storage/jialu/ECG/logdir/logdir-pretrained_vit/best2.ckpt \
        /media/Datacenter_storage/jialu_/jialu_causalv2/logdir/logdir_pretrain_vit_new/best3.ckpt \
        --model "causal" --logtofile True
        ;;
    3)
        echo "Running Conf..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json \
        MCAT_vit_losssave_conf \
        --num_workers 8 --device_ids "3" \
        --pre_train_backbone \
        /media/Datacenter_storage/jialu/003/logdir/logdir-pretrain_vit_conf_ecg/best2.ckpt \
        /media/Datacenter_storage/jialu/003/logdir/logdir_pretrain_vit_conf_cxr/best3.ckpt \
        --model "main" --logtofile True
        ;;
    4)
        echo "Running CaConf..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json \
        /MCAT_vit_losssave_caconf \
        --num_workers 8 --device_ids "3" \
        --pre_train_backbone \
        /media/Datacenter_storage/jialu/003/logdir/logdir-pretrain_vit_conf_ecg/best2.ckpt \
        /media/Datacenter_storage/jialu/003/logdir/logdir_pretrain_vit_conf_cxr/best3.ckpt \
        --model "causal" --logtofile True
    
        echo "Invalid choice."
        ;;
esac

exit
