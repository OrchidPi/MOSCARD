# MOSCARD - Multimodal Opportunistic Screening for Cardiovascular Adverse events with Causal Reasoning and De-confounding
Our study addresses bias in multimodal medical imaging by integrating causal reasoning techniques. We utilize chest X-ray (CXR) images as the primary source of information and employ electrocardiogram (ECG) signals as a complementary guiding modality. To effectively preserve and leverage the essential features from CXR images while incorporating insights from ECG data, we have adapted a [co-attention mechanism](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file) originally developed for processing H&E stained whole slide images alongside genomic factors. For single modality training, we employ a Vision Transformer (ViT) architecture, specifically utilizing the [MedCLIP](https://github.com/RyanWangZf/MedCLIP) image modality, to serve as a unified encoder for both ECG signals and CXR images during the encoder training phase. This integration allows for a cohesive and comprehensive analysis of the multimodal medical data.

### Model framework and Causal graph 

* Model overview:
<img src="https://github.com/OrchidPi/MOSCARD/blob/main/examples/model.png" width="100%" align="middle"/>
Proposed MOSCARD architecture and de-confounding causal reasoning graph, input X, task label Y, causal factor A, confounder C, directed edges for causal confounder relations: (a) Step 1 – single modality encoder training with confusion loss; (b) Step 2 multimodal learning with co-attention and SCM; (c) Step 1 training – Single modality; (d) Step 2 Multimodal training with co-attention and causal intervention.

### Performance: Comparative model performance after multimodal data alignment with combined and individual modality (95% confidence intervals using bootstrapping).  

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
    <td>MACE_6M</td>
    <td>0.653[0.652, 0.655]</td><td>0.711[0.709, 0.712]</td>
    <td>0.643[0.642, 0.645]</td><td>0.690[0.688, 0.690]</td>
    <td>0.639[0.637, 0.641]</td><td>0.681[0.678, 0.682]</td>
  </tr>
  <tr>
    <td>MACE_1yr</td>
    <td>0.663[0.663, 0.665]</td><td>0.725[0.723, 0.726]</td>
    <td>0.651[0.649, 0.653]</td><td>0.705[0.703, 0.708]</td>
    <td>0.646[0.645, 0.649]</td><td>0.688[0.685, 0.691]</td>
  </tr>
  <tr>
    <td>MACE_2yr</td>
    <td>0.656[0.653, 0.658]</td><td>0.717[0.716, 0.719]</td>
    <td>0.647[0.645, 0.648]</td><td>0.702[0.698, 0.704]</td>
    <td>0.646[0.646, 0.649]</td><td>0.690[0.687, 0.691]</td>
  </tr>
  <tr>
    <td>MACE_5yr</td>
    <td>0.653[0.651, 0.655]</td><td>0.712[0.710, 0.715]</td>
    <td>0.640[0.635, 0.640]</td><td>0.698[0.696, 0.699]</td>
    <td>0.642[0.641, 0.644]</td><td>0.687[0.684, 0.690]</td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td>
    <td>0.634[0.614, 0.640]</td><td>0.662[0.654, 0.679]</td>
    <td>0.606[0.590, 0.616]</td><td>0.630[0.623, 0.653]</td>
    <td>0.634[0.617, 0.640]</td><td>0.711[0.692, 0.724]</td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td>
    <td>0.715[0.708, 0.715]</td><td>0.792[0.789, 0.796]</td>
    <td>0.739[0.729, 0.737]</td><td>0.804[0.800, 0.809]</td>
    <td>0.672[0.668, 0.678]</td><td>0.737[0.733, 0.742]</td>
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
    <td>MACE_6M</td>
    <td>0.652[0.649, 0.653]</td><td>0.711[0.707, 0.711]</td>
    <td>0.644[0.643, 0.647]</td><td>0.695[0.692, 0.696]</td>
    <td>0.641[0.640, 0.645]</td><td>0.686[0.684, 0.687]</td>
  </tr>
  <tr>
    <td>MACE_1yr</td>
    <td>0.662[0.661, 0.663]</td><td>0.724[0.723, 0.726]</td>
    <td>0.656[0.653, 0.658]</td><td>0.713[0.711, 0.714]</td>
    <td>0.647[0.645, 0.648]</td><td>0.694[0.693, 0.696]</td>
  </tr>
  <tr>
    <td>MACE_2yr</td>
    <td>0.654[0.653, 0.656]</td><td>0.715[0.714, 0.718]</td>
    <td>0.648[0.646, 0.649]</td><td>0.706[0.704, 0.708]</td>
    <td>0.649[0.649, 0.652]</td><td>0.695[0.695, 0.698]</td>
  </tr>
  <tr>
    <td>MACE_5yr</td>
    <td>0.650[0.647, 0.651]</td><td>0.711[0.708, 0.711]</td>
    <td>0.647[0.645, 0.649]</td><td>0.703[0.701, 0.704]</td>
    <td>0.642[0.640, 0.644]</td><td>0.687[0.685, 0.688]</td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td>
    <td>0.669[0.658, 0.677]</td><td>0.677[0.662, 0.690]</td>
    <td>0.623[0.619, 0.642]</td><td>0.638[0.620, 0.649]</td>
    <td>0.640[0.627, 0.650]</td><td>0.710[0.685, 0.713]</td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td>
    <td>0.737[0.734, 0.741]</td><td>0.810[0.806, 0.813]</td>
    <td>0.772[0.767, 0.773]</td><td>0.837[0.833, 0.839]</td>
    <td>0.673[0.671, 0.680]</td><td>0.753[0.748, 0.761]</td>
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
    <td>MACE_6M</td>
    <td>0.681[0.681, 0.684]</td><td>0.737[0.736, 0.739]</td>
    <td>0.671[0.671, 0.673]</td><td>0.722[0.720, 0.724]</td>
    <td>0.666[0.665, 0.668]</td><td>0.721[0.719, 0.722]</td>
  </tr>
  <tr>
    <td>MACE_1yr</td>
    <td>0.691[0.690, 0.693]</td><td>0.751[0.750, 0.752]</td>
    <td>0.678[0.675, 0.678]</td><td>0.737[0.736, 0.741]</td>
    <td>0.671[0.669, 0.672]</td><td>0.730[0.728, 0.731]</td>
  </tr>
  <tr>
    <td>MACE_2yr</td>
    <td>0.683[0.680, 0.687]</td><td>0.745[0.743, 0.746]</td>
    <td>0.672[0.671, 0.674]</td><td>0.730[0.730, 0.735]</td>
    <td>0.673[0.672, 0.675]</td><td>0.732[0.731, 0.736]</td>
  </tr>
  <tr>
    <td>MACE_5yr</td>
    <td>0.679[0.678, 0.681]</td><td>0.740[0.739, 0.742]</td>
    <td>0.664[0.661, 0.665]</td><td>0.723[0.721, 0.725]</td>
    <td>0.672[0.672, 0.674]</td><td>0.727[0.724, 0.727]</td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td>
    <td>0.623[0.619, 0.641]</td><td>0.673[0.658, 0.683]</td>
    <td>0.594[0.587, 0.609]</td><td>0.649[0.634, 0.668]</td>
    <td>0.663[0.647, 0.670]</td><td>0.649[0.632, 0.667]</td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td>
    <td>0.701[0.697, 0.707]</td><td>0.777[0.773, 0.782]</td>
    <td>0.728[0.722, 0.728]</td><td>0.789[0.779, 0.787]</td>
    <td>0.635[0.635, 0.643]</td><td>0.693[0.687, 0.696]</td>
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
    <td>MACE_6M</td>
    <td>0.681[0.679, 0.682]</td><td>0.733[0.732, 0.735]</td>
    <td>0.670[0.668, 0.670]</td><td>0.716[0.715, 0.718]</td>
    <td>0.673[0.671, 0.674]</td><td>0.723[0.722, 0.725]</td>
  </tr>
  <tr>
    <td>MACE_1yr</td>
    <td>0.691[0.689, 0.692]</td><td>0.750[0.749, 0.752]</td>
    <td>0.681[0.676, 0.683]</td><td>0.734[0.733, 0.736]</td>
    <td>0.674[0.672, 0.676]</td><td>0.735[0.734, 0.738]</td>
  </tr>
  <tr>
    <td>MACE_2yr</td>
    <td>0.683[0.681, 0.684]</td><td>0.740[0.738, 0.742]</td>
    <td>0.669[0.665, 0.670]</td><td>0.723[0.723, 0.726]</td>
    <td>0.675[0.673, 0.676]</td><td>0.733[0.731, 0.734]</td>
  </tr>
  <tr>
    <td>MACE_5yr</td>
    <td>0.678[0.677, 0.679]</td><td>0.735[0.733, 0.738]</td>
    <td>0.669[0.667, 0.670]</td><td>0.717[0.716, 0.719]</td>
    <td>0.675[0.673, 0.676]</td><td>0.730[0.729, 0.735]</td>
  </tr>
  <tr>
    <td>External datasets (MIMIC)</td>
    <td>MACE_6M</td>
    <td>0.623[0.614, 0.635]</td><td>0.662[0.649, 0.676]</td>
    <td>0.600[0.590, 0.615]</td><td>0.644[0.624, 0.658]</td>
    <td>0.611[0.594, 0.619]</td><td>0.657[0.633, 0.670]</td>
  </tr>
  <tr>
    <td>External datasets (ED)</td>
    <td>MACE_1yr</td>
    <td>0.702[0.697, 0.704]</td><td>0.777[0.768, 0.783]</td>
    <td>0.732[0.727, 0.734]</td><td>0.809[0.799, 0.811]</td>
    <td>0.647[0.645, 0.653]</td><td>0.710[0.705, 0.714]</td>
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

  - ALBEF Baseline training reference is from [code].


* Model Testing
  - Single modality testing
    - CXR single modality: `bash CXR/scrpits/train.sh`
    - ECG single modality: `bash ECG/scrpits/train.sh`
  - MOSCARD training testing: `bash MOSCARD/scrpits/test.sh`
  - MedCLIP Baseline testing: `bash MedClip_baseline/scripts/test.sh`
  - ALBEF Baseline testing reference is from [code].

* Model Weights
  - Model weights for the proposed MOSCARD model, including Step 1: single-modality training weights, and Step 2: multi-modal classification training weights.
    Location: [google drive](https://drive.google.com/drive/folders/10IcvmM1VtWpMg3cK7XkLnVyc-KuK433V?usp=sharing)


* Saliency map figure
To plot a saliency map, you can refer to the [code]([https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb)).

### Contact
* If you have any quesions, please post it on github issues or email me.

### Reference
* [https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file)
* [https://github.com/RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)
* [https://github.com/dy1901/ecg_plot/tree/master](https://github.com/dy1901/ecg_plot/tree/master)
* [https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb)


