#!/bin/bash
echo "Running MedClip Baseline CXR+ECG..."
PYTHONPATH=.:$PYTHONPATH python MedClip_baseline/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Combined' --out_csv_path 'test/MedClip/mimic_combined.csv' --device_ids "3" 

echo "Running MedClip Baseline CXR..."
PYTHONPATH=.:$PYTHONPATH python MedClip_baseline/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'CXR' --out_csv_path 'test/MedClip/mimic_CXR.csv' --device_ids "3" 

echo "Running MedClip Baseline ECG..."
PYTHONPATH=.:$PYTHONPATH python MedClip_baseline/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'ECG' --out_csv_path 'test/MedClip/mimic_ECG.csv' --device_ids "3" 
