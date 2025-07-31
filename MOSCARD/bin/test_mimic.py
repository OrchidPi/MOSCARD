from asyncio import subprocess
import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import pandas as pd
import scipy.stats as st
import random
import re  # Import regex module
from tqdm import tqdm
import gcsfs  # Required to read from GCS

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset_mimic_test import ImageDataset_Mayo_bimodal #ImageDataset_bimodal  
from model.MOSCARD import coatt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./', type=str, help="Path to the trained models")
    parser.add_argument('--in_csv_path', default='/media/Datacenter_storage/jialu/003/mimic_ECG_view_images/mimic_test_modify.csv', type=str, help="Path to the input image path in csv")
    parser.add_argument('--test_model', default='Baseline', type=str, help="Test model name [Baseline, Conf, Causal, CaConf]")
    parser.add_argument('--out_csv_path', default='test/mimic_test.csv', type=str, help="Path to the output predictions in csv")
    parser.add_argument('--num_workers', default=8, type=int, help="Number of workers for each data loader")
    parser.add_argument('--device_ids', default='0,1,2,3', type=str, help="GPU indices, comma-separated (e.g., '0,1')")
    args = parser.parse_args()
    return args


if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    """Get predictions from model output."""
    if cfg.criterion_target in ['BCE', "FL"]:
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion: {}'.format(cfg.criterion_target))
    return pred


def test_epoch(cfg, device_ids, model, dataloader, out_csv_path):
    """Run inference and save results to CSV."""
    torch.set_grad_enabled(False)
    model.eval()
    #device_ids = list(map(int, device_ids.split(',')))
    # TODO: make this a self.device and refactor as a class
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)
    print(f"Number of tasks: {num_tasks}")
    # Define correct column order
    pred_cols = ["MACE_6M", "MACE_1yr", "MACE_2yr", "MACE_5yr"]
    combined_pred = [f"combined_pred_{x}" for x in pred_cols]
    CXR_pred = [f"CXR_pred_{x}" for x in pred_cols]
    ECG_pred = [f"ECG_pred_{x}" for x in pred_cols]

    MACE_labels = ["MACE_6M", "MACE_1yr", "MACE_2yr", "MACE_5yr"]

   
    test_header = ["img_path1", "img_path2"] + combined_pred + CXR_pred + ECG_pred + MACE_labels

    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        print(f"Test header written: {test_header}")
        for step in tqdm(range(steps), desc="Model Test Starting", unit="batch", ncols=80):
            #image1, image2, path1, path2, labels = next(dataiter)
            print(f"Step: {step}")
            index, patient, image1, image2, path1, path2, labels = next(dataiter)
            image1 = image1.to(device)
            image2 = image2.to(device)
            # print(f"image1:{image1}, image2:{image2}")

            _, combined, CXR_output, ECG_output, _, _, _, _ = model(image1, image2)

            batch_size = len(path1)

            # Get predictions for combined, CXR, and ECG outputs
            combined_pred = np.zeros((num_tasks, batch_size))
            CXR_pred = np.zeros((num_tasks, batch_size))
            ECG_pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                print(f"Task: {i}")
                combined_pred[i,:] = get_pred(combined[i], cfg)
                CXR_pred[i,:] = get_pred(CXR_output[i], cfg)
                ECG_pred[i,:] = get_pred(ECG_output[i], cfg)

            for i in range(batch_size):
                print(f"Batch: {i}")
                combined_batch = ','.join(map(lambda x: '{}'.format(x),  combined_pred[:, i]))
                CXR_batch = ','.join(map(lambda x: '{}'.format(x),  CXR_pred[:, i]))
                ECG_batch = ','.join(map(lambda x: '{}'.format(x),  ECG_pred[:, i]))
                MACE_label = ','.join(map(lambda x: '{}'.format(x),  labels[i]))

                result = f"{path1[i]},{path2[i]},{combined_batch},{CXR_batch},{ECG_batch},{MACE_label}"
                f.write(result + '\n')
    print(f"Test results written to: {out_csv_path}")
    # saving CSV to GCS
    gcs_path = f"gs://moscard-data-98b3/static/mimics-forty-five/output-data/{os.path.basename(out_csv_path)}"
    save_csv_to_gcs(out_csv_path, gcs_path)

def test_epoch_df(cfg, device_ids, model, dataloader):
    """
    Run inference and return the results as a DataFrame (no tqdm).
    """
    torch.set_grad_enabled(False)
    model.eval()
    # device handling --------------------------------------------------------
    if isinstance(device_ids, str):
        device_ids = list(map(int, device_ids.split(',')))
    device = torch.device(
        f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[test_epoch_df] Using device -> {device}")
    # column names -----------------------------------------------------------
    pred_cols  = ["MACE_6M", "MACE_1yr", "MACE_2yr", "MACE_5yr"]
    comb_cols  = [f"combined_pred_{c}" for c in pred_cols]
    cxr_cols   = [f"CXR_pred_{c}"      for c in pred_cols]
    ecg_cols   = [f"ECG_pred_{c}"      for c in pred_cols]
    label_cols = pred_cols
    header     = ["img_path1", "img_path2"] + comb_cols + cxr_cols + ecg_cols + label_cols
    # storage for rows -------------------------------------------------------
    rows = []
    steps = len(dataloader)
    num_tasks = len(cfg.num_classes)
    print(f"[test_epoch_df] Total batches: {steps} | Tasks per sample: {num_tasks}")
    # -----------------------------------------------------------------------
    for step, batch in enumerate(dataloader, start=1):
        print(f"[test_epoch_df] Processing batch {step}/{steps}")
        # ---- unpack --------------------------------------------------------
        index, patient, image1, image2, path1, path2, labels = batch
        image1, image2 = image1.to(device), image2.to(device)
        # ---- forward pass --------------------------------------------------
        _, combined, cxr_out, ecg_out, *_ = model(image1, image2)

        batch_size = image1.size(0)

        # ---- predictions ---------------------------------------------------
        combined_pred = np.zeros((num_tasks, batch_size))
        cxr_pred      = np.zeros_like(combined_pred)
        ecg_pred      = np.zeros_like(combined_pred)

        for t in range(num_tasks):
            print(f"[test_epoch_df] Processing task {t+1}/{num_tasks}")
            combined_pred[t] = get_pred(combined[t], cfg)
            cxr_pred[t]      = get_pred(cxr_out[t],  cfg)
            ecg_pred[t]      = get_pred(ecg_out[t],  cfg)

        labels_np = labels.numpy()  # (batch, tasks)

        # ---- build row per sample -----------------------------------------
        for i in range(batch_size):
            print(f"[test_epoch_df] Processing sample {i+1}/{batch_size}")
            row = {
                "img_path1": path1[i],
                "img_path2": path2[i],
            }

            # predictions
            for t, col in enumerate(comb_cols):
                row[col] = combined_pred[t, i]
            for t, col in enumerate(cxr_cols):
                row[col] = cxr_pred[t, i]
            for t, col in enumerate(ecg_cols):
                row[col] = ecg_pred[t, i]

            # labels
            for t, col in enumerate(label_cols):
                row[col] = labels_np[i, t]

            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    print(f"[test_epoch_df] Finished. DataFrame shape: {df.shape}")
    return df

def save_csv_to_gcs(local_path, gcs_path):
    """Saves a local file to Google Cloud Storage."""
    try:
        # Assumes you have authenticated with 'gcloud auth application-default login'
        # or have the necessary environment variables set for authentication.
        fs = gcsfs.GCSFileSystem()
        fs.put(local_path, gcs_path)
        print(f"Successfully uploaded {local_path} to {gcs_path}")
    except Exception as e:
        print(f"Failed to upload {local_path} to {gcs_path}: {e}")

def extract_numeric(value):
    """Extract numeric value from tensor-like strings."""
    if isinstance(value, str) and "tensor" in value:  # Check if value is a tensor string
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)  # Extract numeric part
        if match:
            return int(float(match.group()))  # Convert to integer (0 or 1)
    return int(float(value))  # Convert normal numeric values

def calculate_metrics(csv_path, pred_col):
    """Calculate AUC and Accuracy for a given prediction column."""
    df_pre = pd.read_csv(csv_path)

    # Convert MACE_1yr to integers (handle tensor string format)
    df_pre['MACE_6M'] = df_pre['MACE_6M'].apply(extract_numeric)

    # Compute ROC curve & AUC
    fpr, tpr, thresholds = metrics.roc_curve(df_pre['MACE_6M'], df_pre[pred_col], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Compute Optimal Threshold
    optimal_threshold = Find_Optimal_Cutoff(df_pre['MACE_6M'].tolist(), df_pre[pred_col].tolist())[0]

    # Convert predictions to binary based on threshold
    binary_preds = (np.array(df_pre[pred_col].tolist()) >= optimal_threshold).astype(int)

    # Compute accuracy
    binary_accuracy = np.mean(binary_preds == df_pre['MACE_6M'])


    # Bootstrap Confidence Interval for AUC
    AUC_bootstrap = []
    for i in range(100):
        no = random.randrange(df_pre.shape[0], 20, 3)
        temp = df_pre.sample(n=no, replace=True)
        AUC_bootstrap.append(metrics.roc_auc_score(temp['MACE_6M'], temp[pred_col]))

    AUC_low, AUC_high = st.t.interval(0.95, len(AUC_bootstrap) - 1, loc=np.mean(AUC_bootstrap), scale=st.sem(AUC_bootstrap))

    # Bootstrap Confidence Interval for Accuracy
    ACC_bootstrap = []
    for i in range(100):
        no = random.randrange(df_pre.shape[0], 20, 3)
        temp = df_pre.sample(n=no, replace=True)
        temp_preds = (temp[pred_col] >= optimal_threshold).astype(int)
        ACC_bootstrap.append(np.mean(temp_preds == temp['MACE_6M']))

    ACC_low, ACC_high = st.t.interval(0.95, len(ACC_bootstrap) - 1, loc=np.mean(ACC_bootstrap), scale=st.sem(ACC_bootstrap))

    # Print results
    print(f"{pred_col}:")
    print(f"  - AUC: {auc:.3f} (95% CI: [{AUC_low:.3f}, {AUC_high:.3f}])")
    print(f"  - Accuracy: {binary_accuracy:.3f} (95% CI: [{ACC_low:.3f}, {ACC_high:.3f}])\n")

    # print(f"{pred_col}: AUC = {auc:.3f}, Accuracy = {binary_accuracy:.3f}")

    return auc, binary_accuracy



def Find_Optimal_Cutoff(target, predicted):
    """Find the optimal probability cutoff for classification."""
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def load_checkpoint_from_gcs(gcs_path, device):
    """
    Loads a model checkpoint from GCS using gcsfs and returns the checkpoint.
    """
    fs = gcsfs.GCSFileSystem()
    """with fs.open(gcs_path, 'rb') as f:
        ckpt = torch.load(f, map_location=device)"""
    with fs.open(gcs_path, "rb") as f:
        ckpt = torch.load(f, map_location=device, weights_only=False)
    return ckpt

def run_GCP(in_csv, config_path="MOSCARD/config/config.json", test_model="Baseline", 
            out_csv_path='test/mimic_test.csv', 
            device_ids='0,1,2,3', num_workers=8):
    print("Running test_mimic.run_GCP...") 
    print(f"Config path: {config_path}, Test model: {test_model}, Output CSV path: {out_csv_path}")
    print(f"Device IDs: {device_ids}, Number of workers: {num_workers}")
    # Load configuration
    if not os.path.exists(config_path):
        #result = subprocess.run(["ls", "-l"], capture_output=True, text=True)
        print("Configuration path does not exist. Here are the available items:")
        print(f"Current directory: {os.getcwd()}")
        for item in os.listdir('.'):
            print(f" - {item}")
        #print(result.stdout)  # Output of the command
        #raise FileNotFoundError(f"Configuration path {config_path} does not exist.")
    try:
        with open(config_path) as f:
            print(f"Loading configuration from: {f.name}")
            cfg = edict(json.load(f))
            print("Successfully loaded configuration from:", f.name)
    except Exception as e:
        print(f"Error loading configuration: {e}")

    device_ids = list(map(int, device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    """if num_devices < len(device_ids):
        raise Exception(f"# available GPU: {num_devices} < --device_ids: {len(device_ids)}")"""

    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else "cpu")
    print(f'Is cuda available? {torch.cuda.is_available()}')
    
    model = coatt(cfg)  # MCAT(cfg) if needed
    print(f"coatt created successfully.")
    gcs_ckpt_base = 'gs://moscard-data-98b3/static/model-weights'

    # Determine which model to load
    if test_model == 'Baseline':
        ckpt_path = f'{gcs_ckpt_base}/Baseline.ckpt'
    elif test_model == 'Conf':
        ckpt_path = f'{gcs_ckpt_base}/Conf.ckpt'
    elif test_model == 'Causal':
        ckpt_path = f'{gcs_ckpt_base}/Causal.ckpt'
    elif test_model == 'CaConf':
        ckpt_path = f'{gcs_ckpt_base}/CaConf.ckpt'
    else:
        raise ValueError(f"Unsupported test_model: {test_model}")
    print(f"Loading model from: {ckpt_path}")
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt = load_checkpoint_from_gcs(ckpt_path, device)
    model.module.load_state_dict(ckpt['state_dict'], strict=False)

    print("Model loaded successfully.")

    dataloader_test = DataLoader(
        ImageDataset_Mayo_bimodal(in_csv, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=num_workers,
        drop_last=False, shuffle=False
    )

    #test_epoch(cfg, device_ids, model, dataloader_test, out_csv_path)
    # test_epoch_df(cfg, device_ids, model, dataloader)
    results_df = test_epoch_df(cfg, device_ids, model, dataloader_test)
    # Save results to CSV
    results_df.to_csv(out_csv_path, index=False)
    # print('Save best step:', ckpt['step'], 'AUC:', ckpt['auc_dev_best'])
    print(f"Results saved to: {out_csv_path}")
    gcs_path = f"gs://moscard-data-98b3/static/mimics-forty-five/output-data/{os.path.basename(out_csv_path)}"
    save_csv_to_gcs(out_csv_path, gcs_path)
    # calculate on preexisting data?
    print("Yey I did the thing!")

def run(args):
    print("Running test_mimic...")
    print(args)
    with open(args.model_path + './MOSCARD/config/config.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices > len(device_ids):
        raise Exception(f"# available GPU: {num_devices} < --device_ids: {len(device_ids)}")
    # TODO: make this a self.device and refactor as a class
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else "cpu")
    #device = torch.device(f'cuda:{device_ids[0]}')
    print(f'Is cuda available? {torch.cuda.is_available()}')
    
    if args.test_model == 'Baseline':
        model = coatt(cfg) #MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Baseline.ckpt')
    elif args.test_model == 'Conf':
        model = coatt(cfg) #MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Conf.ckpt')
    elif args.test_model == 'Causal':
        model = coatt(cfg) #MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Causal.ckpt')
    elif args.test_model == 'CaConf':
        model = coatt(cfg) #MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/CaConf.ckpt')
    
    
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.module.load_state_dict(ckpt['state_dict'], strict=False)

    dataloader_test = DataLoader(
        ImageDataset_Mayo_bimodal(args.in_csv_path, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False
    )

    test_epoch(cfg, args, model, dataloader_test, args.out_csv_path)

    # print('Save best step:', ckpt['step'], 'AUC:', ckpt['auc_dev_best'])

    """# Calculate metrics for each prediction type
    for pred_type in ["combined", "CXR", "ECG"]:
        calculate_metrics(args.out_csv_path, f"{pred_type}_pred_MACE_6M")"""


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
