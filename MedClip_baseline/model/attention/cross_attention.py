from torch import nn
import torch
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Define linear projections for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2, modality=None):
        """
        x1: features of CXR images (modality A)
        x2: features of ECG signals (modality B)
        modality: which modality to focus on for query
        """
        batch_size, seq_length1, embed_dim1 = x1.size()
        batch_size, seq_length2, embed_dim2 = x2.size()

        # Ensure both input embeddings have the same dimension
        assert embed_dim1 == self.embed_dim and embed_dim2 == self.embed_dim, \
            "Input embedding dimensions must match the specified embed_dim"

        if modality == "ECG":
            # ECG as query, CXR as key/value
            QY = self.query(x1)
            KY = self.key(x2)
            VY = self.value(x2)
        elif modality == "CXR":
            # CXR as query, ECG as key/value
            QY = self.query(x2)
            KY = self.key(x1)
            VY = self.value(x1)
        else:
            raise ValueError("Modality must be either 'ECG' or 'CXR'")

        # Split heads for multi-head attention
        QY = QY.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        KY = KY.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        VY = VY.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(QY, KY.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, VY)

        # Concatenate heads and project output back to embed_dim
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attention_output)

        return output
