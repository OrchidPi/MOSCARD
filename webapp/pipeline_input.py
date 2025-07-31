import datetime
from flask import Flask
from flask import request, Response
import sys
import json
from types import SimpleNamespace
import os
import subprocess
import google.cloud.storage as storage
import pandas as pd
import logging
import time
logger = logging.basicConfig(level=logging.INFO)

def list_available_packages():
    import pkgutil
    for module in pkgutil.iter_modules():
        print(module.name)

try:
    from MOSCARD.bin import test_mimic
    print("Sucessfully Imported using: from MOSCARD.bin import test_mimic")
except:
    print("Failed to import from MOSCARD.bin import test_mimic")
    print("This may be due to system path getting overwriten or an issue with the module itself")
    print("Uncomment this following line in pipeline_input.py to list the available modules")
    #list_available_packages()
    sys.path.append(os.path.join(os.getcwd(),"MOSCARD/bin"))
    import test_mimic
    print("Sucessfully Imported using: import test_mimic")
    

def get_model_checkpoints_from_google_bucket(bucket_name):
    """Get model checkpoints from a Google Cloud Storage bucket."""
    # requires install: google-cloud-storage
    # moscard-data-98b3/static/model-weights/<checkpoint_name>.ckpt
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix="static/model-weights/")
    return [blob.name for blob in blobs if blob.name.endswith(".ckpt")]

def load_model_checkpoint(bucket_name, checkpoint_name):
    """Load a model checkpoint from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"static/model-weights/{checkpoint_name}.ckpt")
    return blob.download_as_bytes()

def run_test_mimic(runID):
    
    time.sleep(4)
    print("Running pipeline_input.run_test_mimic...")
    checkpoint_list = get_model_checkpoints_from_google_bucket("moscard-data-98b3")
    print("Available checkpoints:")
    for checkpoint in checkpoint_list:
        print(f" - {checkpoint}")
    

    # open csv examples/mimic_test_randy_local.csv to df"""
    """in_df = pd.read_csv("/app/src/examples/mimic_test_randy_local.csv")
    print(in_df.head())"""
    in_csv = ["/app/src/examples/mimic_test_randy_local.csv",
              "examples/mimic_test_randy_local.csv"][1]
    #"MOSCARD/ckpt./MOSCARD/config/config.json"
    test_mimic.run_GCP(in_csv, config_path="MOSCARD/config/config.json", test_model="Baseline",
                       out_csv_path='test/mimic_test.csv',
                       device_ids='0', num_workers=8)

if __name__ == "__main__":
    # sys.argv[0] is the script name, sys.argv[1] is the first argument, etc.
    if len(sys.argv) < 3:
        print("Usage: python pipeline_input.py <step> <runID>")
        sys.exit(1)
    step = sys.argv[1]
    runID = sys.argv[2]
    #run_preprocess(step, runID)
    run_test_mimic("test")