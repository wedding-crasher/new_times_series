import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Attentions import FullAttention, AttentionLayer
import numpy as np


class Model(nn.Module):

    # torch 모든 Layer들은 nn.Module 상속
    def __init__(self, configs):
        super(Model, self).__init__()
        # Module의 생성자 상속으로 실행
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,  # 시계열 길이
            configs.d_model,  # Embedding 차원
            configs.embed,  # Embed 방식?
            configs.freq,  # ?
            configs.dropout,  # Dropout 비율
        )
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.ouput_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
            if self.use_norm:
                means = x_enc.mean(1, keepdim=True).detach()
                x_enc = x_enc - means
                stdev = torch.sqrt(
                    torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x_enc /= stdev

            _, _, N = x_enc.shape

            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            dec_out = self.projector(enc_out).permute(0, 2, 1)[
                :, :, :N
            ]  # filter covariates

            if self.use_norm:
                # De-Normalization from Non-stationary Transformer
                dec_out = dec_out * (
                    stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                )
                dec_out = dec_out + (
                    means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                )

            return dec_out
