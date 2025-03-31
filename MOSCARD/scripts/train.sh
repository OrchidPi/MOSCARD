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
        MOSCARD/config/config.json \
        Multimodal_baseline \
        --num_workers 8 --device_ids "0" \
        --pre_train_backbone \
        ECG/ckpt/ECG.ckpt \
        CXR/ckpt/CXR.ckpt \
        --model "main" --logtofile True
        ;;
    2)
        echo "Running Causal..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        MOSCARD/config/config.json \
        Multimodal_causal \
        --num_workers 8 --device_ids "0" \
        --pre_train_backbone \
        ECG/ckpt/ECG.ckpt \
        CXR/ckpt/CXR.ckpt \
        --model "causal" --logtofile True
        ;;
    3)
        echo "Running Conf..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        MOSCARD/config/config.json \
        Multimodal_conf \
        --num_workers 8 --device_ids "0" \
        --pre_train_backbone \
        ECG/ckpt/ECG_conf.ckpt \
        CXR/ckpt/CXR_conf.ckpt \
        --model "main" --logtofile True
        ;;
    4)
        echo "Running CaConf..."
        PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/train.py \
        MOSCARD/config/config.json \
        Multimodal_caconf \
        --num_workers 8 --device_ids "0" \
        --pre_train_backbone \
        ECG/ckpt/ECG_conf.ckpt \
        CXR/ckpt/CXR_conf.ckpt \
        --model "causal" --logtofile True
    
        echo "Invalid choice."
        ;;
esac

exit
