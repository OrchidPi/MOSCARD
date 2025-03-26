### main

# {

# PYTHONPATH=.:$PYTHONPATH python /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/bin/train_MCAT.py /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json /media/Datacenter_storage/jialu/003/MACE/logdir_MACE_final/MCAT_vit_losssave --num_workers 8 --device_ids "0" --pre_train_backbone /media/Datacenter_storage/jialu/ECG/logdir/logdir-pretrained_vit/best2.ckpt /media/Datacenter_storage/jialu_/jialu_causalv2/logdir/logdir_pretrain_vit_new/best3.ckpt --model "only_main" --logtofile True

#     exit
# }

### causal 

# {

# PYTHONPATH=.:$PYTHONPATH python /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/bin/train_MCAT_causal.py /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json /media/Datacenter_storage/jialu/003/MACE/logdir_MACE/MCAT_vit_losssave_causal --num_workers 8 --device_ids "3" --pre_train_backbone /media/Datacenter_storage/jialu/ECG/logdir/logdir-pretrained_vit/best2.ckpt /media/Datacenter_storage/jialu_/jialu_causalv2/logdir/logdir_pretrain_vit_new/best3.ckpt --model "causal" --logtofile True

#     exit
# }
 


### conf

{

PYTHONPATH=.:$PYTHONPATH python /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/bin/train_MCAT.py /media/Datacenter_storage/jialu/003/MACE/Multimodal_MACE/config/Mayo.json /media/Datacenter_storage/jialu/003/MACE/logdir_MACE_final/MCAT_vit_losssave_conf --num_workers 8 --device_ids "2" --pre_train_backbone /media/Datacenter_storage/jialu/003/logdir/logdir-pretrain_vit_conf_ecg/best2.ckpt /media/Datacenter_storage/jialu/003/logdir/logdir_pretrain_vit_conf_cxr/best3.ckpt --model "only_main" --logtofile True

    exit
}
 