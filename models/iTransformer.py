import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    
    
    # torch 모든 Layer들은 nn.Module 상속
    def __init__(self, configs):
        super(Model, self).__init__()ncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
        # Module의 생성자 상속으로 실행
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, #시계열 길이 
            configs.d_model, # Embedding 차원
            configs.embed,  # Embed 방식? 
            configs.freq, #? 
            configs.dropout, #Dropout 비율
        )
        self.class_strategy = configs.class_strategy
        
        #Encoder-only architecture 
        # self.encoder 