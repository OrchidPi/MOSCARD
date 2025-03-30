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
import re  
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset_mimic_test import ImageDataset_bimodal  
from model.MOSCARD import coatt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./', type=str, help="Path to the trained models")
    parser.add_argument('--in_csv_path', default='examples/mimic_test.csv', type=str, help="Path to the input image path in csv")
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


def test_epoch(cfg, args, model, dataloader, out_csv_path):
    """Run inference and save results to CSV."""
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    pred_cols = ["MACE_6M", "MACE_1yr", "MACE_2yr", "MACE_5yr"]
    combined_pred = [f"combined_pred_{x}" for x in pred_cols]
    CXR_pred = [f"CXR_pred_{x}" for x in pred_cols]
    ECG_pred = [f"ECG_pred_{x}" for x in pred_cols]

    MACE_labels = ["MACE_6M", "MACE_1yr", "MACE_2yr", "MACE_5yr"]

   
    test_header = ["img_path1", "img_path2"] + combined_pred + CXR_pred + ECG_pred + MACE_labels

    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')

        for step in tqdm(range(steps), desc="Model Test Starting", unit="batch", ncols=80):
            index, patient, image1, image2, path1, path2, labels = next(dataiter)
            image1 = image1.to(device)
            image2 = image2.to(device)
            # print(f"image1:{image1}, image2:{image2}")

            _, combined, CXR_output, ECG_output, _, _, _, _ = model(image1, image2)

            batch_size = len(path1)
            combined_pred = np.zeros((num_tasks, batch_size))
            CXR_pred = np.zeros((num_tasks, batch_size))
            ECG_pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                combined_pred[i,:] = get_pred(combined[i], cfg)
                CXR_pred[i,:] = get_pred(CXR_output[i], cfg)
                ECG_pred[i,:] = get_pred(ECG_output[i], cfg)

            for i in range(batch_size):
                combined_batch = ','.join(map(lambda x: '{}'.format(x),  combined_pred[:, i]))
                CXR_batch = ','.join(map(lambda x: '{}'.format(x),  CXR_pred[:, i]))
                ECG_batch = ','.join(map(lambda x: '{}'.format(x),  ECG_pred[:, i]))
                MACE_label = ','.join(map(lambda x: '{}'.format(x),  labels[i]))

                result = f"{path1[i]},{path2[i]},{combined_batch},{CXR_batch},{ECG_batch},{MACE_label}"
                f.write(result + '\n')



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
    df_pre['MACE_6M'] = df_pre['MACE_6M'].apply(extract_numeric)

    # Compute ROC curve & AUC
    fpr, tpr, thresholds = metrics.roc_curve(df_pre['MACE_6M'], df_pre[pred_col], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    optimal_threshold = Find_Optimal_Cutoff(df_pre['MACE_6M'].tolist(), df_pre[pred_col].tolist())[0]
    binary_preds = (np.array(df_pre[pred_col].tolist()) >= optimal_threshold).astype(int)
    # Compute accuracy
    binary_accuracy = np.mean(binary_preds == df_pre['MACE_6M'])


    # Bootstrap Confidence Interval for AUC
    AUC_bootstrap = []
    for i in range(100):
        no = random.randrange(20, df_pre.shape[0], 3)
        temp = df_pre.sample(n=no, replace=True)
        AUC_bootstrap.append(metrics.roc_auc_score(temp['MACE_6M'], temp[pred_col]))

    AUC_low, AUC_high = st.t.interval(0.95, len(AUC_bootstrap) - 1, loc=np.mean(AUC_bootstrap), scale=st.sem(AUC_bootstrap))

    # Bootstrap Confidence Interval for Accuracy
    ACC_bootstrap = []
    for i in range(100):
        no = random.randrange(20, df_pre.shape[0], 3)
        temp = df_pre.sample(n=no, replace=True)
        temp_preds = (temp[pred_col] >= optimal_threshold).astype(int)
        ACC_bootstrap.append(np.mean(temp_preds == temp['MACE_6M']))

    ACC_low, ACC_high = st.t.interval(0.95, len(ACC_bootstrap) - 1, loc=np.mean(ACC_bootstrap), scale=st.sem(ACC_bootstrap))

    # Print results
    print(f"{pred_col}:")
    print(f"  - AUC: {auc:.3f} (95% CI: [{AUC_low:.3f}, {AUC_high:.3f}])")
    print(f"  - Accuracy: {binary_accuracy:.3f} (95% CI: [{ACC_low:.3f}, {ACC_high:.3f}])\n")


    return auc, binary_accuracy



def Find_Optimal_Cutoff(target, predicted):
    """Find the optimal probability cutoff for classification."""
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def run(args):
    with open(args.model_path + './MOSCARD/config/config.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(f"# available GPU: {num_devices} < --device_ids: {len(device_ids)}")
    
    device = torch.device(f'cuda:{device_ids[0]}')

    if args.test_model == 'Baseline':
        model = MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Baseline.ckpt')
    elif args.test_model == 'Conf':
        model = MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Conf.ckpt')
    elif args.test_model == 'Causal':
        model = MCAT(cfg)
        ckpt_path = os.path.join(args.model_path, './MOSCARD/ckpt/Causal.ckpt')
    elif args.test_model == 'CaConf':
        model = MCAT(cfg)
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

    # Calculate metrics for each prediction type
    for pred_type in ["combined", "CXR", "ECG"]:
        calculate_metrics(args.out_csv_path, f"{pred_type}_pred_MACE_6M")


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
