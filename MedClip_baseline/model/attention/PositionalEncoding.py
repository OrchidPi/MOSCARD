import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=128):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length, modality="CXR"):
        super(LearnedPositionalEncoding, self).__init__()
        
        if modality == "ECG":
            self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 1024))  # not generative
            #self.position_embeddings = nn.Parameter(torch.zeros(1, 1024, 512)) # generative
        elif modality == "CXR":
            self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 1024))
            #self.position_embeddings = nn.Parameter(torch.zeros(1, 1024, 512))  
        else:
            raise ValueError(f"Invalid modality: {modality}. Expected 'ECG' or 'CXR'.")

    def forward(self, x, position_ids=None):
        # print("x within leanred positional embedding:", x.shape)
        position_embeddings = self.position_embeddings
        # print("x", x.shape)
        # print("position_embeddings", position_embeddings.shape)
        return x + position_embeddings

