import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


"""
iTransforemr Architecture
Embedding Multivariate Attention -> LayerNorm -> Feed-forward(CONV1D로 구현해도됨, 둘다 선형변환) -> LayerNorm -> Projection (output)
입력형태 -> R= (n,d)차원 
참고로 프로젝트는 전체 레이어에서 한번만있음
"""


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # FeedForward 차원설정
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1
        )  # 선형변환
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1
        )  # 후에 차원 복원
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x,
            x,
            x,  # self attention applied to variate toekns
            attn_mask=attn_mask,  # Optional attention mask
            tau=tau,
            delta=delta,
            # Additional Parameters in attetnion
        )
        x = x + self.dropout(
            new_x
        )  # Residual Connection + dropout -> 앞에 aTTENTION Layer 출력에  Dropout적용
        y = x = self.norm1(x)  # Normalization 적용
        # CONV1입력형식(batchsize, in_channels, in_width) ->Output: (batchsize, out_channels, out_width) 맞추기 위해 transpose해서 다시 projection
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # 두번째 Feedforward 통과 -> 이미 통과하고 다시 차원 되돌리기 위해
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)

        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns