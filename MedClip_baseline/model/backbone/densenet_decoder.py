import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.utils import get_norm

class _UpsampleDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm_type='Unknown'):
        super(_UpsampleDenseLayer, self).__init__()
        self.add_module('norm1', get_norm(norm_type, num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.ConvTranspose2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', get_norm(norm_type, bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.ConvTranspose2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    
    def forward(self, x):
        new_features = super(_UpsampleDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _UpsampleDenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_type='Unknown'):
        super(_UpsampleDenseBlock, self).__init__()
        for i in range(num_layers):
            # Dynamically update input features for each layer
            layer_input_features = num_input_features + i * growth_rate
            layer = _UpsampleDenseLayer(layer_input_features, growth_rate, bn_size, drop_rate, norm_type=norm_type)
            self.add_module('upsample_denselayer%d' % (i + 1), layer)


class _UpsampleTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_type='Unknown'):
        super(_UpsampleTransition, self).__init__()
        self.add_module('norm', get_norm(norm_type, num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.ConvTranspose2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('upsample', nn.ConvTranspose2d(num_output_features, num_output_features, kernel_size=2, stride=2))  # Upsample by factor of 2


class DenseNetDecoder(nn.Module):
    #TODO block size list
    def __init__(self, growth_rate=32, block_config=(16, 24, 12, 6), norm_type='Unknown', num_init_features=1024, bn_size=4, drop_rate=0, num_classes=1):
        super(DenseNetDecoder, self).__init__()

        # Reverse of the final dense block in DenseNet-121
        num_features = num_init_features  # Start with 1024 channels
        self.block1 = nn.Sequential()
        self.trans1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.trans2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.trans3 = nn.Sequential()
        self.block4 = nn.Sequential()

        # Block 1: Reduce from 1024 to 512
        block = _UpsampleDenseBlock(num_layers=block_config[0], num_input_features=num_features, norm_type=norm_type,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.block1.add_module('upsample_denseblock1', block)
        num_features = num_features + block_config[0] * growth_rate  # Update the number of features after the block
        num_features = 1024
        trans = _UpsampleTransition(num_input_features=num_features, num_output_features=num_features // 2, norm_type=norm_type)
        self.trans1.add_module('upsample_transition1', trans)
        num_features = num_features // 2

        # Block 2: Reduce from 512 to 256
        block = _UpsampleDenseBlock(num_layers=block_config[1], num_input_features=num_features, norm_type=norm_type,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.block2.add_module('upsample_denseblock2', block)
        num_features = num_features + block_config[1] * growth_rate  # Update the number of features after the block
        num_features = 512
        trans = _UpsampleTransition(num_input_features=num_features, num_output_features=num_features // 2, norm_type=norm_type)
        self.trans2.add_module('upsample_transition2', trans)
        num_features = num_features // 2


        # Block 3: Reduce from 256 to 128
        block = _UpsampleDenseBlock(num_layers=block_config[2], num_input_features=num_features, norm_type=norm_type,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.block3.add_module('upsample_denseblock3', block)
        num_features = num_features + block_config[2] * growth_rate  # Update the number of features after the block
        num_features = 256
        #num_features = 64
        trans = _UpsampleTransition(num_input_features=num_features, num_output_features=num_features // 2, norm_type=norm_type)
        self.trans3.add_module('upsample_transition3', trans)
        num_features = num_features // 2

        # Block 4: Reduce from 128 to 64
        block = _UpsampleDenseBlock(num_layers=block_config[3], num_input_features=num_features, norm_type=norm_type,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.block4.add_module('upsample_denseblock4', block)
        
        # Make sure this feature size calculation matches the expected input of the next layer
        num_features = num_features + block_config[3] * growth_rate
        num_features = 64
        
        # Final upsampling to reach (N, 3, 512, 512)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),  # No change in size
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # No change in size
        )

    def forward(self, x):
        print(f"before decoder:{x.shape}")
        x = self.block1(x)
        print(f"block1:{x.shape}")
        x = x[:, 512:, :, :]  # 512
        print(f"block1:{x.shape}")
        x = self.trans1(x)
        print(f"trans1:{x.shape}")
        x = self.block2(x)
        x = x[:, 768:, :, :] # 256
        print(f"block2:{x.shape}")
        x = self.trans2(x)
        print(f"trans2:{x.shape}")
        x = self.block3(x)
        x = x[:, 384:, :, :] # 128
        print(f"block3:{x.shape}")
        x = self.trans3(x)
        print(f"trans3:{x.shape}")
        x = self.block4(x)
        x = x[:, 256:, :, :] # 64
        print(f"block4:{x.shape}")
        out = self.final_upsample(x)
        print(f"after decoder:{out.shape}")
        out = torch.sigmoid(out)
        return out

