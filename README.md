# MOSCARD - Multimodal Opportunistic Screening for Cardiovascular Adverse events with Causal Reasoning and De-confounding
Our study addresses bias in multimodal medical imaging by integrating causal reasoning techniques. We utilize chest X-ray (CXR) images as the primary source of information and employ electrocardiogram (ECG) signals as a complementary guiding modality. To effectively preserve and leverage the essential features from CXR images while incorporating insights from ECG data, we have adapted a [co-attention mechanism](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file) originally developed for processing H&E stained whole slide images alongside genomic factors. For single modality training, we employ a Vision Transformer (ViT) architecture, specifically utilizing the [MedCLIP](https://github.com/RyanWangZf/MedCLIP) image modality, to serve as a unified encoder for both ECG signals and CXR images during the encoder training phase. This integration allows for a cohesive and comprehensive analysis of the multimodal medical data.

### Model framework and Causal graph 

* Model overview:
<img src="https://github.com/OrchidPi/MOSCARD/blob/main/examples/model.png" width="100%" align="middle"/>
Proposed MOSCARD architecture and de-confounding causal reasoning graph, input X, task label Y, causal factor A, confounder C, directed edges for causal confounder relations: (a) Step 1 – single modality encoder training with confusion loss; (b) Step 2 multimodal learning with co-attention and SCM; (c) Step 1 training – Single modality; (d) Step 2 Multimodal training with co-attention and causal intervention.

### Performance

<table border="1">
  <tr>
    <th colspan="10">Baseline Multimodal</th>
  </tr>
  <tr>
    <th rowspan="2">Dataset Types</th>
    <th rowspan="2">MACE</th>
    <th colspan="2">CXR+ECG (Combined)</th>
    <th colspan="2">CXR+ECG (CXR)</th>
    <th colspan="2">CXR+ECG (ECG)</th>
  </tr>
  <tr>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
  </tr>
  <tr>
    <td rowspan="4">Internal datasets (PCI)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_2yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_5yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
    <th colspan="10">Causal Multimodal</th>
  </tr>
  <tr>
    <th rowspan="2">Dataset Types</th>
    <th rowspan="2">MACE</th>
    <th colspan="2">CXR+ECG (Combined)</th>
    <th colspan="2">CXR+ECG (CXR)</th>
    <th colspan="2">CXR+ECG (ECG)</th>
  </tr>
  <tr>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
  </tr>
  <tr>
    <td rowspan="4">Internal datasets (PCI)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_2yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_5yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
    <th colspan="10">Conf Multimodal</th>
  </tr>
  <tr>
    <th rowspan="2">Dataset Types</th>
    <th rowspan="2">MACE</th>
    <th colspan="2">CXR+ECG (Combined)</th>
    <th colspan="2">CXR+ECG (CXR)</th>
    <th colspan="2">CXR+ECG (ECG)</th>
  </tr>
  <tr>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
  </tr>
  <tr>
    <td rowspan="4">Internal datasets (PCI)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_2yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_5yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
    <th colspan="10">CaConf Multimodal</th>
  </tr>
  <tr>
    <th rowspan="2">Dataset Types</th>
    <th rowspan="2">MACE</th>
    <th colspan="2">CXR+ECG (Combined)</th>
    <th colspan="2">CXR+ECG (CXR)</th>
    <th colspan="2">CXR+ECG (ECG)</th>
  </tr>
  <tr>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
    <th>Accuracy</th><th>AUC</th>
  </tr>
  <tr>
    <td rowspan="4">Internal datasets (PCI)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_2yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>MACE_5yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td><td></td><td></td><td></td><td></td><td></td><td></td>
  </tr>
</table>



### Train the models

* Data preparation
  - Prepare your data by following the example provided in `config/train.csv`.
  - Update the data path in `config/config.json`.
  - Convert ECG signals into image representations using the [ecg_plot library](https://github.com/dy1901/ecg_plot/tree/master)
  - Delect all the lateral images of chest X-ray datasets by runing example command: `python CXR/view_clf/inference.py`

* Model Training
Ensure all necessary packages are installed by running:
`pip install -r requirements.txt`
  - Single modality training
    - CXR single modality Baseline follow the on-screen prompt to choose a mode (1: Baseline single modality encoder training; 2: Single modality encoder training with confusion loss): `bash CXR/scrpits/train.sh`
    - ECG single modality Baseline follow the on-screen prompt to choose a mode (1: Baseline single modality encoder training; 2: Single modality encoder training with confusion loss): `bash ECG/scrpits/train.sh`

  - MOSCARD training
    - Train follow the on-screen prompt to choose a mode (1–4): `bash MOSCARD/scrpits/train.sh`
      
    | Option | Mode Name | Description |
    |--------|-----------|-------------|
    | 1 | Baseline | Trains a baseline multimodal model using pre-trained ECG and CXR backbones without de-confounding and causal reasoning. |
    | 2 | Causal | Trains the model with causal reasoning mechanisms based on Baseline. |
    | 3 | Conf | Trains the single baseline model using backbones that were trained with de-confounding strategies. |
    | 4 | CaConf | Trains the causal model using de-confounding backbones (Final Proposed model). |

  - MedCLIP Baseline training: Train follow the on-screen prompt to choose a mode (1-2): `bash MedClip_baseline/scripts/train.sh`
    - Mode 1 focuses on learning shared representations between modalities through CLIP-based alignment and cross-attention.
    - Mode 2 performs downstream classification by freezing the alignment backbone and training only the final MLP classifiers.

  - ALBEF Baseline training reference is from [code](https://github.com/rimitalahiri92/ALBEF_baselines).


* Model Testing
  - Single modality testing
    - CXR single modality: `bash CXR/scrpits/train.sh`
    - ECG single modality: `bash ECG/scrpits/train.sh`
  - MOSCARD training testing: `bash MOSCARD/scrpits/test.sh`
  - MedCLIP Baseline testing: `bash MedClip_baseline/scripts/test.sh`
  - ALBEF Baseline testing reference is from [code](https://github.com/rimitalahiri92/ALBEF_baselines).

* Model Weights
  - Model weights for the proposed MOSCARD model, including Step 1: single-modality training weights, and Step 2: multi-modal classification training weights.
    Location: [google drive](https://drive.google.com/drive/folders/10IcvmM1VtWpMg3cK7XkLnVyc-KuK433V?usp=sharing)


* Saliency map figure
To plot a saliency map, you can refer to the [code]([https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb)).

### Contact
* If you have any quesions, please post it on github issues or email [me](jialupi@asu.edu)

### Reference
* [https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file)
* [https://github.com/RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)
* [https://github.com/dy1901/ecg_plot/tree/master](https://github.com/dy1901/ecg_plot/tree/master)
* [https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb)


