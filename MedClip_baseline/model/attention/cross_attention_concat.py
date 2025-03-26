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

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        batch_size, seq_length1, embed_dim1 = x1.size()  # [3, 1024, 512]
        batch_size, seq_length2, embed_dim2 = x2.size()  # [3, 1024, 512]
        
        # Ensure the embed dimensions match
        assert embed_dim1 == embed_dim2, "Embedding dimensions of x1 and x2 must match!"
        assert embed_dim1 == self.embed_dim, "Embedding dimension must match model initialization"


        # Stack x1 and x2 together
        Y = torch.cat((x1, x2), dim=1)

        # Linear transformations
        QY = self.query(Y)
        KY = self.key(Y)
        VY = self.value(Y)

        # Split into heads
        QY = QY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)
        KY = KY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)
        VY = VY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(QY, KY.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, VY)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length1 + seq_length2, embed_dim1)

        # Final linear projection
        output = self.out_proj(attention_output)


        return output
