#!/bin/bash
echo "Running Baseline..."
PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Baseline' --out_csv_path 'test/mimic_baseline.csv' --device_ids "3" 

echo "Running Confounder..."
PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Conf' --out_csv_path 'test/mimic_conf.csv' --device_ids "3" 

echo "Running Causal..."
PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'Causal' --out_csv_path 'test/mimic_causal.csv' --device_ids "3" 

echo "Running Causal_confounder..."
PYTHONPATH=.:$PYTHONPATH python MOSCARD/bin/test_mimic.py --in_csv_path 'examples/mimic_test.csv' --test_model 'CaConf' --out_csv_path 'test/mimic_caconf.csv' --device_ids "3" 