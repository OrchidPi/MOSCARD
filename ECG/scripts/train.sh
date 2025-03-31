#!/bin/bash

echo "Select which mode to run:"
echo "1) ECG"
echo "2) ECG confounder"


read -p "Enter your choice [1-2]: " choice

case $choice in
    1)
        echo "Running ECG..."
        PYTHONPATH=.:$PYTHONPATH python ECG/bin/train_ECG.py \
        ECG/config/config.json \
        ECG_single \
        --num_workers 8 --device_ids "0" \
        --logtofile True
        ;;
    2)
        echo "Running ECG confounder..."
        PYTHONPATH=.:$PYTHONPATH python ECG/bin/train_ECG_conf.py \
        ECG/config/config.json \
        ECG_single_conf \
        --num_workers 8 --device_ids "0" \
        --logtofile True


        echo "Invalid choice."
        ;;
esac

exit
