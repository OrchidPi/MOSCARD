#!/bin/bash
echo "Running CXR..."
PYTHONPATH=.:$PYTHONPATH python CXR/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'main' --out_csv_path 'test/CXR/mimic_CXR_baseline.csv' --device_ids "3" 

echo "Running CXR Confounder..."
PYTHONPATH=.:$PYTHONPATH python CXR/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Conf' --out_csv_path 'test/CXR/mimic_CXR_conf.csv' --device_ids "3" 

