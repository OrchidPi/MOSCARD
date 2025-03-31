# MOSCARD - Multimodal Opportunistic Screening for Cardiovascular Adverse events with Causal Reasoning and De-confounding
Our study addresses bias in multimodal medical imaging by integrating causal reasoning techniques. We utilize chest X-ray (CXR) images as the primary source of information and employ electrocardiogram (ECG) signals as a complementary guiding modality. To effectively preserve and leverage the essential features from CXR images while incorporating insights from ECG data, we have adapted a [co-attention mechanism](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file) originally developed for processing H&E stained whole slide images alongside genomic factors. For single modality training, we employ a Vision Transformer (ViT) architecture, specifically utilizing the [MedCLIP](https://github.com/RyanWangZf/MedCLIP) image modality, to serve as a unified encoder for both ECG signals and CXR images during the encoder training phase. This integration allows for a cohesive and comprehensive analysis of the multimodal medical data.

### Model framework and Causal graph 

* Model overview:
<img src="https://github.com/OrchidPi/MOSCARD/blob/main/examples/model.png" width="100%" align="middle"/>
Proposed MOSCARD architecture and de-confounding causal reasoning graph, input X, task label Y, causal factor A, confounder C, directed edges for causal confounder relations: (a) Step 1 - single modality encoder training with confusion loss; (b) Step 2, multimodal learning with co-attention and SCM; (c) Step 1 training – Single modality; (d) Step 2 Multimodal training with co-attention and causal intervention.

### Train the models

* Data preparation
> Prepare your data by following the example provided in `config/train.csv`.
> Update the data path in `config/config.json`.
> Convert ECG signals into image representations using the [ecg_plot library](https://github.com/dy1901/ecg_plot/tree/master)

* Model Training
Ensure all necessary packages are installed by running:
`pip install -r requirements.txt`
> Single modality
>> CXR single modality Baseline : `bash CXR/scrpits/train.sh`
>> ECG single modality Baseline : `bash ECG/scrpits/train.sh`
> Proposed Multimodal MOSCARD
>> `bash MOSCARD/scrpits/train.sh`
>> ### Mode Options

| Option | Mode Name   | Description                                                                 |
|--------|-------------|-----------------------------------------------------------------------------|
| 1      | Baseline    | Trains a baseline multimodal model using pre-trained ECG and CXR backbones without de-confouding and causal reasoning. |
| 2      | Causal      | Trains the model with causal reasoning mechanisms based on Baseline.                   |
| 3      | Conf        | Trains the single baseline model using backbones that were trained with de-confounder strategies. |
| 4      | CaConf      | Trains the causal model using de-confounding backbones.                    |


> Baseline Debiased model : `python bin/train_debiased.py  config/Mayo.json logdir/logdir_debiased --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

> MOSCARD : `python bin/train_pretrained.py  config/Mayo.json logdir/logdir_pretrain --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

> Baseline Debiased model : `python bin/train_debiased.py  config/Mayo.json logdir/logdir_debiased --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

>> 
>> Final Causal+Confounder model (Baseline Causal model+Baseline Confounder model with causal feature concat) : `python bin/train_causalconf.py config/Mayo.json logdir/logdir_causalconf --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

* Model Testing
To test your model, run the example command:
> `python logdir/logdir_causalconf/classification/bin/test_internal.py`

* Grad-CAM figure
To plot a saliency map, you can refer to the [code]([https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb)).

### Contact
* If you have any quesions, please post it on github issues or email [me](jialupi@asu.edu)

### Reference
* [https://github.com/jfhealthcare/Chexpert/tree/master](https://github.com/jfhealthcare/Chexpert/tree/master)
* [https://github.com/zhihengli-UR/DebiAN](https://github.com/zhihengli-UR/DebiAN)
* [https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/adityac94/Grad_CAM_plus_plus/tree/masterhttps://github.com/adityac94/Grad_CAM_plus_plus/tree/master)


