#!/bin/bash
echo "Running ECG..."
PYTHONPATH=.:$PYTHONPATH python ECG/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'main' --out_csv_path 'test/ECG/mimic_ECG_baseline.csv' --device_ids "3" 

echo "Running ECG Confounder..."
PYTHONPATH=.:$PYTHONPATH python ECG/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Conf' --out_csv_path 'test/ECG/mimic_ECG_conf.csv' --device_ids "3" 

