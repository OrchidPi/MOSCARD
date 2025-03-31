# MOSCARD - Multimodal Opportunistic Screening for Cardiovascular Adverse events with Causal Reasoning and De-confounding
This study aims to mitigate bias in medical imaging by incorporating causal reasoning. We leverage a well-performed [pretrained model](https://github.com/jfhealthcare/Chexpert/tree/master) to enhance the robustness and fairness of predictions.

### Model framework and Causal graph 

* Model overview:
<img src="https://github.com/OrchidPi/Mitigating-Bias-in-Medical-imaging-with-Causal-Reasoning/blob/main/config/Causal%20Framework.png" width="100%" align="middle"/>

* Causal graph:
<img src="https://github.com/OrchidPi/Mitigating-Bias-in-Medical-imaging-with-Causal-Reasoning/blob/main/config/Causal%20Graph.png" width="50%" align="middle"/>

### Train the models

* Data preparation
> Prepare your data by following the example provided in `config/train.csv`.
> Update the data path in `config/config.json`.

* Model Training
Ensure all necessary packages are installed by running:
`pip install -r requirements.txt`
> Baseline Pretrained model : `python bin/train_pretrained.py  config/Mayo.json logdir/logdir_pretrain --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

> Baseline Debiased model : `python bin/train_debiased.py  config/Mayo.json logdir/logdir_debiased --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

> Ablation study :
>> Baseline Causal model : `python bin/train_baselinecausal.py  config/Mayo.json logdir/logdir_causal --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`
>> 
>> Baseline Confounder model : `python bin/train_conf.py  config/Mayo.json logdir/logdir_causal --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`
>> 
>> Causal model (Baseline Causal model+causal feature concat): `python bin/train_causal.py  config/Mayo.json logdir/logdir_causal --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`
>> 
>> Causal+Confounder model (Baseline Causal model+Baseline Confounder model) : `python bin/train_causalconf_noconcat.py config/Mayo.json logdir/logdir_causalconf_noconcat --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`
>> 
>> Final Causal+Confounder model (Baseline Causal model+Baseline Confounder model with causal feature concat) : `python bin/train_causalconf.py config/Mayo.json logdir/logdir_causalconf --num_workers 8 --device_ids "0,1"  --pre_train "config/pre_train.pth"  --logtofile True`

* Model Testing
To test your model, run the example command:
> `python logdir/logdir_causalconf/classification/bin/test_internal.py`

* Grad-CAM figure
To plot a Grad-CAM figure, you can refer to the [code from this GitHub repository](https://github.com/adityac94/Grad_CAM_plus_plus/tree/master).

### Contact
* If you have any quesions, please post it on github issues or email [me](jialupi@asu.edu)

### Reference
* [https://github.com/jfhealthcare/Chexpert/tree/master](https://github.com/jfhealthcare/Chexpert/tree/master)
* [https://github.com/zhihengli-UR/DebiAN](https://github.com/zhihengli-UR/DebiAN)
* [https://github.com/adityac94/Grad_CAM_plus_plus/tree/master](https://github.com/adityac94/Grad_CAM_plus_plus/tree/masterhttps://github.com/adityac94/Grad_CAM_plus_plus/tree/master)


