from torch import nn
import torch
import numpy as np

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet_crossattention import (densenet121, densenet169, densenet201, create_twomodal_densenet)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap
from model.attention.cross_attention import CrossAttention 
from model.attention.transformer_ import Transformer



BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Mlp(nn.Module):
    """ Multilayer perceptron with minimal feature distortion."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()  # Activation layer
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

        # Use LayerNorm for minimal distribution changes (optional)
        self.norm1 = nn.LayerNorm([hidden_features, 1, 1])  # Applies to spatial dimensions
        self.norm2 = nn.LayerNorm([out_features, 1, 1])

        # Initialize a residual connection
        self.residual = nn.Conv2d(in_features, out_features, kernel_size=1) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)  # Save the original input as residual
        x = self.fc1(x)  # First linear transformation
        x = self.norm1(x)  # Normalize
        x = self.act(x)  # Apply activation
        x = self.drop(x)  # Dropout
        x = self.fc2(x)  # Second linear transformation
        x = self.norm2(x)  # Normalize
        x = self.drop(x)  # Dropout
        x = x + residual  # Add residual connection
        return x

    

class Mlp_projection(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        #self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        #self.fc2 = nn.Linear(hidden_features, out_features)
        #self.norm2 = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)       
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.norm2(x)
        x = self.drop(x)
        return x
    


class Classifier(nn.Module):

    def __init__(self, cfg, pretrained = False):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = create_twomodal_densenet(cfg)
        self.global_pool_CXR = GlobalPool(cfg, pre_train_paths=None, modality="CXR")
        self.global_pool_ECG = GlobalPool(cfg, pre_train_paths=None, modality="ECG")
        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3
        
        self._init_classifier_conf()
        # self._init_classifier_main()
        # self._init_classifier_causal()
        self._init_classifier()
        self._init_classifier_causal_old()
        self._init_bn()
        self._init_bn_causal()
        self._init_attention_map_CXR(pre_train_paths=None, modality="CXR")
        self._init_attention_map_ECG(pre_train_paths=None, modality="ECG")


        img_dim = 16
        patch_dim = 4
        num_channels = int(self.backbone.densenet1.num_features)  # Assuming RGB images
        embed_dim = int(self.backbone.densenet1.num_features)  # Dimension of the embedding space
        num_heads = int(cfg.num_heads)  # Number of attention heads
        ff_dim = int(self.backbone.densenet1.num_features)  # Feed-forward hidden layer size
        dropout_rate = 0
        output_dim = 256


        self.mlp = Mlp(in_features=4096, hidden_features=2048, out_features=1024)
        self.mlp_single = Mlp(in_features=2048, hidden_features=1024, out_features=1024)
        

        self.self_attention_cxr = Transformer(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            positional_encoding_type="learned",
            modality="CXR",
            conv_patch_representation=True
        )

        self.self_attention_ecg = Transformer(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            positional_encoding_type="learned",
            modality="ECG",
            conv_patch_representation=True
        )

        # Define the linear projection layer
        self.projection_cxr = Mlp_projection(in_features=embed_dim, hidden_features=int(embed_dim/2), out_features=int(embed_dim/4))
        self.projection_ecg = Mlp_projection(in_features=embed_dim, hidden_features=int(embed_dim/2), out_features=int(embed_dim/4))

        # Initialize cross-attention module
        self.cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Initialize layers for processing intermediate features
        self.bn_intermediate = nn.BatchNorm1d(1024)
        self.relu_intermediate = nn.ReLU(inplace=True)

        # Freeze the backbone, self-attention, and cross-attention layers if pretrained is True
        if pretrained:
            self._freeze_layers()


    def _freeze_layers(self):
        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.global_pool_CXR.parameters():
            param.requires_grad = False

        for param in self.global_pool_ECG.parameters():
            param.requires_grad = False

        
        # Freeze all parameters in the attention map layers
        if hasattr(self, 'attention_map_CXR') and self.attention_map_CXR is not None:
            for param in self.attention_map_CXR.parameters():
                param.requires_grad = True
        if hasattr(self, 'attention_map_ECG') and self.attention_map_ECG is not None:
            for param in self.attention_map_ECG.parameters():
                param.requires_grad = True


        # Freeze BatchNorm layers initialized in _init_bn
        for index in range(len(self.cfg.num_classes)):
            bn_layer = getattr(self, f"bn_{index}", None)
            if bn_layer is not None:
                bn_layer.eval()  # Set BatchNorm to evaluation mode
                for param in bn_layer.parameters():
                    param.requires_grad = True  # Disable gradient updates

        # Freeze BatchNorm layers initialized in _init_bn_causal
        for index in range(len(self.cfg.num_causal)):
            bn_causal_layer = getattr(self, f"bn_causal_{index}", None)
            if bn_causal_layer is not None:
                bn_causal_layer.eval()  # Set BatchNorm to evaluation mode
                for param in bn_causal_layer.parameters():
                    param.requires_grad = True  # Disable gradient updates



        # Freeze all parameters in the self-attention and cross-attention layers
        for param in self.self_attention_cxr.parameters():
            param.requires_grad = False
        for param in self.self_attention_ecg.parameters():
            param.requires_grad = False
        # for param in self.projection_cxr.parameters():
        #     param.requires_grad = True   
        # for param in self.projection_ecg.parameters():
        #     param.requires_grad = True   
        for param in self.cross_attention.parameters():
            param.requires_grad = False



    
    def _init_classifier_conf(self):
        for index, num_class in enumerate(self.cfg.num_conf):
            classifier = nn.Linear(self.backbone.densenet1.num_features, num_class)
            setattr(self, f"fc_conf_{index}", classifier)

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        1024,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        1024 *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

    def _init_classifier_causal_old(self):
        for index, num_class in enumerate(self.cfg.num_causal):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_causal_old" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_causal_old" +
                    str(index),
                    nn.Conv2d(
                        1024,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_causal_old" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_causal_old" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn_causal(self):
        for index, num_class in enumerate(self.cfg.num_causal):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_causal_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_causal_" +
                    str(index),
                    nn.BatchNorm2d(
                        1024 *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_causal_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )
            



    # def _init_classifier_main(self):
    #     for index, num_class in enumerate(self.cfg.num_classes):
    #         # Replace Conv2d with Linear layers
    #         if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
    #             in_features = 512 * self.expand  # Adjust based on your architecture
    #         elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
    #             in_features = 64
    #         elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
    #             in_features = 2048 * self.expand
    #         else:
    #             raise Exception(f'Unknown backbone type: {self.cfg.backbone}')

    #         setattr(
    #             self,
    #             f"fc_final_{index}",
    #             nn.Sequential(
    #                 nn.Linear(in_features, num_class, bias=True),
    #                 nn.Sigmoid()  # Apply sigmoid to each output
    #             )
    #         )

    #         classifier = getattr(self, f"fc_final_{index}")
    #         if isinstance(classifier[0], nn.Linear):
    #             classifier[0].weight.data.normal_(0, 0.01)
    #             classifier[0].bias.data.zero_()

    
    # def _init_classifier_causal(self):
    #     for index, num_class in enumerate(self.cfg.num_causal):
    #         # Replace Conv2d with Linear layers
    #         if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
    #             in_features = 512 * self.expand  
    #         elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
    #             in_features = 64
    #         elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
    #             in_features = 2048 * self.expand
    #         else:
    #             raise Exception(f'Unknown backbone type: {self.cfg.backbone}')

    #         setattr(
    #             self,
    #             f"fc_causal_{index}",
    #             nn.Sequential(
    #                 nn.Linear(in_features, num_class, bias=True),
    #                 nn.Sigmoid()  # Apply sigmoid to each output
    #             )
    #         )

    #         classifier_causal = getattr(self, f"fc_causal_{index}")
    #         if isinstance(classifier_causal[0], nn.Linear):
    #             classifier_causal[0].weight.data.normal_(0, 0.01)
    #             classifier_causal[0].bias.data.zero_()


    def _init_attention_map_CXR(self, pre_train_paths=None, modality="CXR"):
        # Check for the backbone type and initialize AttentionMap with appropriate parameters
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            self.attention_map_CXR = AttentionMap(
                cfg=self.cfg, 
                num_channels=512, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            self.attention_map_CXR = AttentionMap(
                cfg=self.cfg, 
                num_channels=1024, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            self.attention_map_CXR = AttentionMap(
                cfg=self.cfg, 
                num_channels=2048, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        else:
            raise Exception(f'Unknown backbone type: {self.cfg.backbone}')

    def _init_attention_map_ECG(self, pre_train_paths=None, modality="ECG"):
        # Check for the backbone type and initialize AttentionMap with appropriate parameters
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            self.attention_map_ECG = AttentionMap(
                cfg=self.cfg, 
                num_channels=512, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            self.attention_map_ECG = AttentionMap(
                cfg=self.cfg, 
                num_channels=1024, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            self.attention_map_ECG = AttentionMap(
                cfg=self.cfg, 
                num_channels=2048, 
                pre_train_paths=pre_train_paths, 
                modality=modality
            )
        else:
            raise Exception(f'Unknown backbone type: {self.cfg.backbone}')



    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    
    def forward(self, x1, x2):
        # (N, C, H, W) (N,1024,16,16)
        feat_map1, feat_map2, interm_feat1, interm_feat2 = self.backbone(x1, x2)
        print(f"feat_map1 max:{feat_map1[0].max()},feat_map1 min:{feat_map1[0].min()}" )
        print(f"feat_map2 max:{feat_map2[0].max()},feat_map2 min:{feat_map2[0].min()}" )
        causal_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_causal]
        main_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_classes]
        CXR_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_classes]
        ECG_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_classes]
        conf_CXR_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_conf]
        conf_ECG_logits = [torch.randn(self.cfg.train_batch_size, num) for num in self.cfg.num_conf]

        ### CXR intermediate features (Deconfounding)
        # Process intermediate features
        interm_feat_pooled1 = F.adaptive_avg_pool2d(interm_feat1, (1, 1)).view(interm_feat1.size(0), -1)
        interm_feat_processed1 = self.bn_intermediate(interm_feat_pooled1)
        interm_feat_processed1 = self.relu_intermediate(interm_feat_processed1)

        for index in range(len(self.cfg.num_conf)):
            classifier = getattr(self, f"fc_conf_{index}")
            logit = classifier(interm_feat_processed1)
            conf_CXR_logits[index] = logit.to(interm_feat1.device)

        ### ECG intermediate features (Deconfounding)
        # Process intermediate features
        interm_feat_pooled2 = F.adaptive_avg_pool2d(interm_feat2, (1, 1)).view(interm_feat2.size(0), -1)
        interm_feat_processed2 = self.bn_intermediate(interm_feat_pooled2)
        interm_feat_processed2 = self.relu_intermediate(interm_feat_processed2)

        for index in range(len(self.cfg.num_conf)):
            classifier = getattr(self, f"fc_conf_{index}")
            logit = classifier(interm_feat_processed2)
            conf_ECG_logits[index] = logit.to(interm_feat2.device)
        
        N, C, H, W = feat_map1.size()  # get the dimensions

    
        # Apply self-attention
        feat_map1_attention, old_feat1 = self.self_attention_cxr(feat_map1)
        feat_map2_attention, old_feat2 = self.self_attention_ecg(feat_map2)
        #print(f"after self-attention shape check CXR:{feat_map1_attention.shape}, space check ECG:{feat_map2_attention.shape}")
        # after self-attention shape check CXR:torch.Size([14, 256, 1024]), space check ECG:torch.Size([14, 256, 1024])

        feat_map1_attention_ = torch.mean(feat_map1_attention, dim=1)
        feat_map2_attention_ = torch.mean(feat_map2_attention, dim=1)
        print(f"after self-attention space check CXR:{feat_map1_attention_[0].max()}, space check CXR:{feat_map1_attention_[0].min()}")
        print(f"after self-attention space check ECG:{feat_map2_attention_[0].max()}, space check ECG:{feat_map2_attention_[0].min()}")
        # after self-attention space check CXR:0.9412147402763367, space check ECG:-0.9517126083374023
        
        # Project to reduced dimensions
        # feat_map1_proj = self.projection_cxr(feat_map1_attention)  # Shape: (N, 256, 256)
        # feat_map2_proj = self.projection_ecg(feat_map2_attention)  # Shape: (N, 256, 256)

        #print(f"after linear projection shape check CXR:{feat_map1_proj.shape}, space check ECG:{feat_map2_proj.shape}")


        # Apply cross-attention
        feat_CXR = self.cross_attention(feat_map1_attention, feat_map2_attention, old_feat1, old_feat2, modality="CXR")
        feat_ECG = self.cross_attention(feat_map1_attention, feat_map2_attention, old_feat1, old_feat2, modality="ECG")
        #print(f"after cross-attention shape check CXR:{feat_CXR.shape}, space check ECG:{feat_ECG.shape}")
        # after cross-attention shape check CXR:torch.Size([14, 256, 1024]), space check ECG:torch.Size([14, 256, 1024])

        feat_CXR_ = feat_CXR.mean(dim=1)  # Shape: (N, 1024)
        feat_ECG_ = feat_ECG.mean(dim=1)  # Shape: (N, 1024)
        print(f"after cross-attention space check CXR:{feat_CXR_[0].max()}, space check CXR:{feat_CXR_[0].min()}")
        print(f"after cross-attention space check ECG:{feat_ECG_[0].max()}, space check ECG:{feat_ECG_[0].min()}")

        # after cross-attention space check CXR:0.27231574058532715, space check ECG:-0.31835293769836426


        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                feat_map1 = self.attention_map_CXR(feat_map1)
                print(f"feat_map1 attention shape:{feat_map1.shape},feat_map max:{feat_map1[0].max()}, feat_map1 min:{feat_map1[0].min()}")

            # classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map1 = None
            # (N, C, 1, 1)
            feat1 = self.global_pool_CXR(feat_map1, logit_map1)
            print(f"feat1 global_pool shape:{feat1.shape},feat1 max:{feat1[0].max()}, feat1 min:{feat1[0].min()}")

            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat1 = bn(feat1)
            feat1 = F.dropout(feat1, p=self.cfg.fc_drop, training=self.training)
            print(f"feat1 shape:{feat1.shape},feat1 max:{feat1[0].max()}, feat1 min:{feat1[0].min()}")
            




        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                feat_map2 = self.attention_map_ECG(feat_map2)
                print(f"feat_map2 attention shape:{feat_map2.shape},feat_map2 max:{feat_map2[0].max()}, feat_map2 min:{feat_map2[0].min()}")

            # classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map2 = None
            # (N, C, 1, 1)
            feat2 = self.global_pool_ECG(feat_map2, logit_map2)
            print(f"feat2 global_pool shape:{feat2.shape},feat2 max:{feat2[0].max()}, feat2 min:{feat2[0].min()}")

            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat2 = bn(feat2)
            feat2 = F.dropout(feat2, p=self.cfg.fc_drop, training=self.training)
            print(f"feat2 shape:{feat2.shape},feat2 max:{feat2[0].max()}, feat2 min:{feat2[0].min()}")


        
        for index, num_class in enumerate(self.cfg.num_causal):     
            if self.cfg.attention_map != "None":
                feat_map1 = self.attention_map_CXR(feat_map1)

            # classifier_causal = getattr(self, "fc_causal_" + str(index))
            # (N, 1, H, W)
            logit_map1 = None
            # (N, C, 1, 1)
            feat1_causal = self.global_pool_CXR(feat_map1, logit_map1)

            if self.cfg.fc_bn:
                bn_causal = getattr(self, "bn_causal_" + str(index))
                feat1 = bn_causal(feat1_causal)
            feat1_causal = F.dropout(feat1_causal, p=self.cfg.fc_drop, training=self.training)


        for index, num_class in enumerate(self.cfg.num_causal):     
            if self.cfg.attention_map != "None":
                feat_map2 = self.attention_map_ECG(feat_map2)

            # classifier_causal = getattr(self, "fc_causal_" + str(index))
            # (N, 1, H, W)
            logit_map2 = None
            # (N, C, 1, 1)
            feat2_causal = self.global_pool_ECG(feat_map2, logit_map2)

            if self.cfg.fc_bn:
                bn_causal = getattr(self, "bn_causal_" + str(index))
                feat2 = bn_causal(feat2_causal)
            feat2_causal = F.dropout(feat2_causal, p=self.cfg.fc_drop, training=self.training)


        ### CLIP
        # feat_CXR_1 = feat_CXR_ / feat_CXR_.norm(dim=1, keepdim=True)
        # feat_ECG_1 = feat_ECG_ / feat_ECG_.norm(dim=1, keepdim=True)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * feat_CXR_1 @ feat_ECG_1.t()
        # logits_per_text = logits_per_image.t()

      
        # n = logits_per_image.size(0)
        # labels = torch.arange(n).to(logits_per_image.device)  # Ensure labels are on the same device as logits

        # # # Compute cross-entropy loss for image-to-text (axis=0 in the pseudocode)
        # loss_i = F.cross_entropy(logits_per_image, labels)

        # # # Compute cross-entropy loss for text-to-image (axis=1 in the pseudocode)
        # loss_t = F.cross_entropy(logits_per_text, labels)

        # # # Compute the final loss as the average of loss_i and loss_t
        # loss = (loss_i + loss_t) / 2



        ##  Classification task
        #feat_map = torch.cat([feat_CXR, feat_ECG], dim=2)  # [N, 1024, 512]
        reduce_conv = nn.Conv2d(2048, 1024, kernel_size=1).to(feat1.device)
        reduced_feat1 = reduce_conv(feat1)  # Shape: (N, 1024, 1, 1)
        reduced_feat2 = reduce_conv(feat2)  # Shape: (N, 1024, 1, 1)
        reduced_feat1_causal = reduce_conv(feat1_causal)  # Shape: (N, 1024, 1, 1)
        reduced_feat2_causal = reduce_conv(feat2_causal)  # Shape: (N, 1024, 1, 1)
        feat_CXR_reshaped = feat_CXR_.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1024, 1, 1)
        feat_ECG_reshaped = feat_CXR_.unsqueeze(-1).unsqueeze(-1)  # Shape: (N, 1024, 1, 1)
        combined_feat_CXR = torch.cat([feat_CXR_reshaped, reduced_feat1], dim=1)  # Shape: (N, 2048, 1, 1)
        combined_feat_ECG = torch.cat([feat_ECG_reshaped, reduced_feat2], dim=1)  # Shape: (N, 2048, 1, 1)
        combined_feat = torch.cat([combined_feat_CXR, combined_feat_ECG], dim=1)  # Shape: (N, 4096, 1, 1)
        combined_feat_CXR_causal = torch.cat([feat_CXR_reshaped, reduced_feat1_causal], dim=1)  # Shape: (N, 2048, 1, 1)
        combined_feat_ECG_causal = torch.cat([feat_ECG_reshaped, reduced_feat2_causal], dim=1)  # Shape: (N, 2048, 1, 1)
        combined_feat_causal = torch.cat([combined_feat_CXR_causal, combined_feat_ECG_causal], dim=1)  # Shape: (N, 4096, 1, 1)
        print(f"combined_feat CXR shape before mlp:{combined_feat.shape},combined_feat max:{combined_feat[0].max()}, combined_feat min:{combined_feat[0].min()}")
        print(f"combined_feat ECG shape before mlp:{combined_feat_causal.shape},combined_feat max:{combined_feat_causal[0].max()}, combined_feat min:{combined_feat_causal[0].min()}")
        
        combined_feat = self.mlp(combined_feat)
        combined_feat_causal = self.mlp(combined_feat_causal)
        combined_feat_CXR = self.mlp_single(combined_feat_CXR)
        combined_feat_ECG = self.mlp_single(combined_feat_ECG)
        
        print(f"combined_feat CXR shape after mlp:{combined_feat.shape},combined_feat max:{combined_feat[0].max()}, combined_feat min:{combined_feat[0].min()}")
        print(f"combined_feat ECG shape after mlp:{combined_feat_causal.shape},combined_feat max:{combined_feat_causal[0].max()}, combined_feat min:{combined_feat_causal[0].min()}")
        


        #feat_map = feat_CXR
        # MLP for processing sequence embeddings
        #feat_map = torch.mean(feat_map, dim=1)

        N, _, _, _ = combined_feat.shape
        print(f"feat_map after cross attention:{combined_feat.shape}")
        
        # # Reshape to apply MLP on each sequence element independently (or in parallel)
        # feat_map = self.mlp(feat_map)  # Output shape: (N, 1024, out_features)
        
        # # Pooling across the sequence dimension (1024) to reduce it
        # #feat_map = torch.mean(feat_map, dim=1)  # Shape: (N, out_features)
        # print(f"feat_map after linear layers:{feat_map.shape}") # (32,64)
      


        mf_list = []
        total_mf_list = []
        # feat_map_causal = combined_feat_causal




        # for index, num_class in enumerate(self.cfg.num_causal):
        #     classifier_causal = getattr(self, f"fc_causal_{index}")
            
        #     feat_map_causal = F.dropout(feat_map_causal, p=self.cfg.fc_drop, training=self.training)

        #     # Forward pass through the linear layer and sigmoid
        #     logit = classifier_causal(feat_map_causal)  # Shape: (N, num_class)

        #     # Store the logits
        #     causal_logits[index] = logit.to(feat_map.device)

        #     # Optional: Apply Gumbel softmax if needed
        #     m_f = F.gumbel_softmax(logit, tau=1, hard=True, dim=1)
        #     mf_list.append(m_f)

        #     total_mf_list.append(m_f.view(N, num_class, 1, 1))
        # # Store in total_mf at the correct slice
        # #total_mf[:, current_channel:current_channel + num_class, :, :] = m_f.view(batch_size, num_class, 1, 1)
        # #current_channel += num_class
        # #total_mf = total_mf.to(feat_map.device)

        # total_mf = torch.cat(total_mf_list, dim=1)
        # #print(f"total_mf final shape: {total_mf.shape}")

    
        # for index, num_class in enumerate(self.cfg.num_classes):
        #     classifier = getattr(self, f"fc_final_{index}")
        #     feat_map = F.dropout(feat_map, p=self.cfg.fc_drop, training=self.training)

        #     # Forward pass through the linear layer and sigmoid
        #     logit = classifier(feat_map)  # Shape: (N, num_class)

        #     # Store the logits
        #     main_logits[index] = logit.to(feat_map.device)


        for index, num_class in enumerate(self.cfg.num_causal): 
            classifier_causal = getattr(self, "fc_causal_old" + str(index))    
            logit = classifier_causal(combined_feat_causal)
            #print(f"logit:{logit.shape}")
            # (N, num_class)
            #m_f = self.gumbel_softmax(logit, temperature=0.5, hard=True)
            m_f = F.gumbel_softmax(logit, tau=1, hard=True, dim=1)
            
            mf_list.append(m_f)  # Store each pooled feature

            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_causal:{logit.shape}")

            causal_logits[index] = logit
            #causal_logits.append(logit)

        
        for index, num_class in enumerate(self.cfg.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit = classifier(combined_feat)
            
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_main:{logit.shape}")
            
            main_logits[index] = logit

        for index, num_class in enumerate(self.cfg.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit = classifier(combined_feat_CXR)
            
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_main:{logit.shape}")
            
            CXR_logits[index] = logit
        

        for index, num_class in enumerate(self.cfg.num_classes):
            classifier = getattr(self, "fc_" + str(index))
            logit = classifier(combined_feat_ECG)
            
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            #print(f"logit_main:{logit.shape}")
            
            ECG_logits[index] = logit

    



        return (causal_logits, main_logits, conf_CXR_logits, conf_ECG_logits, CXR_logits, ECG_logits)
