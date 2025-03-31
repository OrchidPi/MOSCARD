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


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Primary device

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset_mimic_test import ImageDataset_test# noqa
from model.classifier_vit import VIT
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                        type=str, help="Path to the trained models")
    parser.add_argument('--in_csv_path', default='mimic_test.csv', metavar='IN_CSV_PATH',
                        type=str, help="Path to the input image path in csv")
    parser.add_argument('--test_model', default='conf', metavar='TEST_MODEL',
                        type=str, help="Test model name [main, conf]")
    parser.add_argument('--out_csv_path', default='test/mimic_test.csv',
                        metavar='OUT_CSV_PATH', type=str,
                        help="Path to the ouput predictions in csv")
    parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                        "workers for each data loader")
    parser.add_argument('--device_ids', default='0,1,2,3', type=str, help="GPU indices "
                        "comma separated, e.g. '0,1' ")
    args = parser.parse_args()
    return args

if not os.path.exists('test'):
    os.mkdir('test')

def get_pred(output, cfg):
    if cfg.criterion_target == 'BCE' or cfg.criterion_target == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion_target))
    return pred



def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)
    num_bias = len(cfg.num_conf)

    task_names = ['MACE']

    pred_6M = ['pred_' + task + '_6M' for task in task_names]
    pred_1yr = ['pred_' + task + '_1yr' for task in task_names]
    pred_2yr = ['pred_' + task + '_2yr' for task in task_names]
    pred_5yr = ['pred_' + task + '_5yr' for task in task_names]


    MACE_6M = [task + '_6M' for task in task_names]
    MACE_1yr = [task + '_1yr' for task in task_names]
    MACE_2yr = [task + '_2yr' for task in task_names]
    MACE_5yr = [task + '_5yr' for task in task_names]


    
    test_header = ['img_path'] + pred_6M + pred_1yr + pred_2yr + pred_5yr + MACE_6M + MACE_1yr + MACE_2yr + MACE_5yr

   
####The out_csv_path file columns should be: img_path, pred_MACE_6M, pred_MACE_1yr,	pred_MACE_2yr,pred_MACE_5yr,MACE
####The column MACE is for target label. The columns pre_MACE_time are the predication for MACE at different time.
    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')

        for step in range(steps):
            image, path, labels = next(dataiter)
            image = image.to(device)
            if args.test_model == 'main':
                output = model(image)
            elif args.test_model == 'conf':
                output, _ = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)

            for i in range(batch_size):
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                MACE_label = ','.join(map(lambda x: '{}'.format(x), labels[i]))
                result = path[i] + ',' + batch + ',' + MACE_label
                f.write(result + '\n')
                logging.info('{}, Image : {}, Prob : {},  MACE_label : {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[i], batch, MACE_label))
                

### Accuracy and AUC calculation:
                
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def calculate_metrics(csv_path):
    df_pre = pd.read_csv(csv_path)
    diseases = ['MACE']
    disease_dfs = {}
    for disease in diseases:
        label_disease = f'{disease}_6M'
        prob_disease = f'pred_{disease}_6M'
        columns = [label_disease, prob_disease]
        disease_dfs[disease] = df_pre[columns]
    df_6M = disease_dfs['MACE']

    fpr, tpr, thresholds = metrics.roc_curve(df_6M['MACE_6M'].tolist(), df_6M['pred_MACE_6M'].tolist(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc:.3f}")

    optimal_threshold = Find_Optimal_Cutoff(df_6M['MACE_6M'].tolist(), df_6M['pred_MACE_6M'].tolist())[0]
    print(f"Optimal Threshold: {optimal_threshold:.3f}")

    binary_preds = (np.array(df_6M['pred_MACE_6M'].tolist()) >= optimal_threshold).astype(int)
    binary_accuracy = np.mean(binary_preds == df_6M['MACE_6M'].tolist())
    print(f"Accuracy: {binary_accuracy:.3f}")

    # Bootstrap for AUC Confidence Interval
    AUC = []
    for i in range(100):
        no = random.randrange(20, df_6M.shape[0], 3)
        temp = df_6M.sample(n=no, replace=True)

        # Check if at least two classes are present in the sampled data
        if len(temp['MACE_6M'].unique()) < 2:
            continue  # Skip this sample if only one class is present

        AUC.append(metrics.roc_auc_score(temp['MACE_6M'], temp['pred_MACE_6M']))

    # Ensure AUC list is not empty before calculating confidence intervals
    if len(AUC) > 0:
        AUC_low, AUC_high = st.t.interval(0.95, len(AUC) - 1, loc=np.mean(AUC), scale=st.sem(AUC))
        print(f"AUC Confidence Interval: [{AUC_low:.3f}, {AUC_high:.3f}]")
    else:
        print("AUC Confidence Interval could not be calculated due to insufficient class diversity in bootstrap samples.")


    # Bootstrap for Accuracy Confidence Interval
    ACC = []
    for i in range(100):
        no = random.randrange(20, df_6M.shape[0], 3)
        temp = df_6M.sample(n=no, replace=True)
        temp_preds = (temp['pred_MACE_6M'] >= optimal_threshold).astype(int)
        ACC.append(np.mean(temp_preds == temp['MACE_6M']))
    ACC_low, ACC_high = st.t.interval(0.95, len(ACC) - 1, loc=np.mean(ACC), scale=st.sem(ACC))
    print(f"Accuracy Confidence Interval: [{ACC_low:.3f}, {ACC_high:.3f}]")





def run(args):
    with open(args.model_path + 'cfg.json') as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    
    if args.test_model == 'main':
        model = VIT(cfg)
        ckpt_path = os.path.join(args.model_path, './CXR/ckpt/CXR.ckpt')
    elif args.test_model == 'conf':
        model = VIT(cfg)
        ckpt_path = os.path.join(args.model_path, './CXR/ckpt/CXR_conf.ckpt')
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.module.load_state_dict(ckpt['state_dict'], strict=False)

    dataloader_test = DataLoader(
        ImageDataset_test(args.in_csv_path, cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    test_epoch(cfg, args, model, dataloader_test, args.out_csv_path)

    print('Save best is step :', ckpt['step'], 'AUC :', ckpt['auc_dev_best'])
    calculate_metrics(args.out_csv_path)



def main():
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
