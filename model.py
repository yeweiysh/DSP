import math
import torch
import torch.nn as nn
from torch.nn import Transformer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False).to(x.device)
        return self.dropout(x)
    
    
class SP_Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(SP_Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.predictor = nn.Linear(in_features=d_model, out_features=vocab_size)
        
    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.empty(size=tokens.size(), dtype=torch.bool)
        key_padding_mask[tokens == 3] = True
        key_padding_mask[tokens != 3] = False
        return key_padding_mask
        
    def forward(self, src, tgt):
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)
        
        src = self.embedding(src)
        
        tgt = self.embedding(tgt)
        
        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask.to(src.device),
            src_key_padding_mask=src_key_padding_mask.to(src.device),
            tgt_key_padding_mask=tgt_key_padding_mask.to(src.device)
        )
        
        return output