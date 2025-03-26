import torch
import torch.nn as nn
import os
from model.attention.self_attention import MultiHeadSelfAttention, TransformerLayer, FeedForward
from model.attention.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        img_dim,  # Image dimension (e.g., 256)
        patch_dim,  # Patch dimension (e.g., 16)
        num_channels,  # Number of input channels (e.g., 3 or 1024)
        embedding_dim,  # Embedding dimension (e.g., 128)
        num_heads,  # Number of heads in multi-head attention (e.g., 8)
        ff_dim,  # Feed-forward dimension in transformer layers (e.g., 512)
        dropout_rate=0.0,  # Dropout rate (default: 0.1)
        positional_encoding_type="learned",  # Type of positional encoding: 'learned' or 'fixed'
        modality="CXR",
        conv_patch_representation=True, 
        pre_train_paths = None
    ):
        super(Transformer, self).__init__()

        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert img_dim % patch_dim == 0, "Image dimension must be divisible by patch dimension"

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.num_patches = int((img_dim // patch_dim) ** 1)  # Total number of patches
        self.seq_length = self.num_patches  # Sequence length is the number of patches

        self.flatten_dim = num_channels  # Flatten dimension of each patch

        # Linear layer to embed the patches
        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)

        # Positional Encoding (either learned or fixed)
        if modality == "CXR":
            if positional_encoding_type == "learned":
                self.position_encoding = LearnedPositionalEncoding(
                    max_position_embeddings=self.seq_length,  # 256 patches
                    embedding_dim=self.embedding_dim,
                    seq_length=self.seq_length, modality="CXR"
                )
            elif positional_encoding_type == "fixed":
                self.position_encoding = FixedPositionalEncoding(
                    self.embedding_dim,
                    max_length=self.seq_length
                )

        elif modality == "ECG":
            if positional_encoding_type == "learned":
                self.position_encoding = LearnedPositionalEncoding(
                    max_position_embeddings=self.seq_length,  # 256 patches
                    embedding_dim=self.embedding_dim,
                    seq_length=self.seq_length, modality="ECG"
                )
            elif positional_encoding_type == "fixed":
                self.position_encoding = FixedPositionalEncoding(
                    self.embedding_dim,
                    max_length=self.seq_length
                )
        else:
            raise ValueError(f"Invalid modality: {modality}. Expected 'ECG' or 'CXR'.")

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        # Transformer Layers (stacked)
        self.layers = TransformerLayer(embed_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim)
        if conv_patch_representation:

            self.conv_x = nn.Conv2d(
                32,
                self.embedding_dim,
                kernel_size=3,
                stride=2,
                padding=1
            )

        # Load pretrained weights if available
        if pre_train_paths and os.path.exists(pre_train_paths):
            state_dict = torch.load(pre_train_paths)
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and v.size() == self.state_dict()[k].size()}
            self.load_state_dict(filtered_state_dict, strict=False)


    def forward(self, x, conv_patch_representation=True):
        if conv_patch_representation:
            # combine embedding with conv patch distribution
            # print("x before patch:", x.shape) # ecg: [8, 1024, 16, 16] cxr: [8, 512, 32, 32] 
            # x = self.bn(x)
            # x = self.relu(x)
            # x = self.conv_x(x)
            # print("x before patch:", x.shape) # ecg: [32, 512, 16, 16] cxr: [32, 512, 16, 16] 
            x = x.permute(0, 2, 3, 1).contiguous()
            # print("x after permute:", x.shape) # ([32, 16, 16, 1024]) ([32, 16, 16, 1024])
            x = x.view(x.size(0), -1, self.embedding_dim) #([16, 32, 32, 512]) ([16, 32, 32, 512])
            # print("conv respresentation is true:", x.shape) # ([32, 256, 1024]) ([32, 256, 1024])

        else:
            # Split the image into patches
            N, C, H, W = x.shape # ecg: [8, 1024, 16, 16] cxr: [8, 512, 32, 32] 
            assert H == self.img_dim and W == self.img_dim, "Image size must match img_dim"
            # Apply unfold to split into patches
            x = (
                x.unfold(2, 2, 2)  # Extract 2x2 windows along the 3rd dimension (height)
                .unfold(3, 2, 2)   # Extract 2x2 windows along the 4th dimension (width)
                #.unfold(4, 2, 2)   # Extract 2x2 windows along the 5th dimension (depth/channel, if it exists)
                .contiguous()      # Ensure the tensor is stored contiguously in memory
            )
            print("x shape after unfold:", x.shape) # [2, 1024, 8, 8, 1, 2, 2]  [8, 512, 16, 16, 1, 2, 2]
            x = x.view(x.size(0), x.size(1), -1, 8) 
            print(x.shape) # [2, 1024, 32, 8] [8, 512, 128, 8]
            x = x.permute(0, 2, 3, 1).contiguous() 
            print(x.shape) # [2, 32, 8, 1024] [8, 128, 8, 512]
            x = x.view(x.size(0), -1, self.flatten_dim)
            print("x shape view:", x.shape) # [2, 256, 1024] [8, 1024, 512] 
            x = self.linear_encoding(x)
            print("linear transformation:", x.shape) # # [2, 256, 1024] [8, 1024, 512]
        
        x = self.position_encoding(x)  # Add positional encodings to the patch embeddings
        # print(f"positional encoding:{x.shape}")
        x = self.pe_dropout(x)
        x = self.layers(x)

        return x
        # x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim)
        # print(f"unfold:{x.shape}")
        # x = x.contiguous().view(N, C, -1, self.patch_dim, self.patch_dim)  # (N, C, num_patches, patch_dim, patch_dim)
        # print(f"patch size:{x.shape}")

        # # Flatten patches into sequences
        # x = x.permute(0, 2, 1, 3, 4)  # (N, num_patches, C, patch_dim, patch_dim)
        # print(f"reshape before flatten:{x.shape}")
        # x = x.contiguous().view(N, self.seq_length, -1)  # (N, num_patches, patch_dim * patch_dim * C)
        # print(f"flatten:{x.shape}")

        # # Apply linear transformation to get embeddings
        # x = self.linear_encoding(x)  # (N, seq_length, embedding_dim)
        # print(f"linear transformation:{x.shape}")

        # # Apply positional encoding
        # x = self.position_encoding(x)  # Add positional encodings to the patch embeddings
        # print(f"positional encoding:{x.shape}")
        # x = self.pe_dropout(x)

        # # Pass through the transformer layers
        # x = self.layers(x)

        # return x
