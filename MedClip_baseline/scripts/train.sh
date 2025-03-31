#!/bin/bash

echo "Select which mode to run:"
echo "1) Alignment"
echo "2) Classifier"


read -p "Enter your choice [1-2]: " choice

case $choice in
    1)
        echo "Running MedClip Baseline Classify..."
        PYTHONPATH=.:$PYTHONPATH python MedClip_baseline/bin/trian.py 
        MedClip_baseline/config/config.json MedClip_Classify 
        --num_workers 8 --device_ids "0,1" 
        --pre_train MedClip_baseline/ckpt/alignment.ckpt
        --model "main" --model_step "Classifier"--logtofile True
        ;;
    2)
        echo "Running MedClip Baseline Classify..."
        PYTHONPATH=.:$PYTHONPATH python MedClip_baseline/bin/trian.py 
        MedClip_baseline/config/config.json MedClip_alignment 
        --num_workers 8 --device_ids "0,1" 
        --pre_train_backbone MedClip_baseline/ckpt/CXR.ckpt MedClip_baseline/ckpt/ECG.ckpt 
        --model "main" --model_step "Classifier"--logtofile True

        echo "Invalid choice."
        ;;
esac

exit
