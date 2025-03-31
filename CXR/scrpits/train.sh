#!/bin/bash

echo "Select which mode to run:"
echo "1) CXR"
echo "2) CXR confounder"


read -p "Enter your choice [1-2]: " choice

case $choice in
    1)
        echo "Running CXG..."
        PYTHONPATH=.:$PYTHONPATH python CXR/bin/train_CXR.py \
        CXR/config/config.json \
        CXR_single \
        --num_workers 8 --device_ids "0" \
        --logtofile True
        ;;
    2)
        echo "Running CXR confounder..."
        PYTHONPATH=.:$PYTHONPATH python CXR/bin/train_CXR_conf.py \
        CXR/config/config.json \
        CXR_single_conf \
        --num_workers 8 --device_ids "0" \
        --logtofile True


        echo "Invalid choice."
        ;;
esac

exit
