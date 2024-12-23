import torch 
import torch.nn as nn
import torch.functional as F 
from math import sqrt 
from utils.masking import TriangularCausalMask
import numpy as np


class FullAttention(nn.Module):
    def __init__(self, mask_flag = True, factor = 5, scale = None, attention_dropout = 0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # B: 배치사이즈, L 시퀀스 길이(토큰개수), H 헤드 수, E 쿼리의 임베딩 차원(각 토큰의 차원)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        # 스케일은 어텐션 스코어가 너무 크거나 작아지는 것을 방지해, 소프트맥스 계산시 더 안정적인 확률분포를 만듬 
        scale = self.scale or 1. / sqrt(E)
        
        #아인슈타인 표기법으로 텐서간 연산, 입력차원 -> 출력차원
        # ex) [B,3,1,4] [B,3,1,4] -> [B,1,3,3]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B,L,device = queries.device)
            
            # mask가 true인 위치에 -np.inf 채워 넣겠다. 
            scores.masked_fill_(attn_mask.mask, -np.inf)
         
        #:[B,H,L,S] dropout 적용해 어텐션 가중치를 무작위로 제거해 과적합 방지 -> 일부값 0으로 설정, 학습시마다 일부 연결 끊어주는 효과 
        A = self.dropout(torch.softmax(scale * scores, dim= -1))
        
        #A와 values 곱해 최종 attention출력 만들기 
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        # Attention map(attention weight) 반환 옵션 
        if self.output_attention:
            return (V.contiguous(),A)
        else:
            return (V.contiguous(), None)
    
        

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys = None, d_values = None):
        super(AttentionLayer,self).__init__()
        # Multi-head Attention이기에 d_model // n_heads로 각 해드별 차원 분할 -> 헤드는 특정정보에 집중
        # 나중에 짜피 CONCAT
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or(d_model // n_heads)

        self.inner_attention = attention

        #queries, keys, values를 각각의 차원으로 projection
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        #최종 출력 projection 레이어: 모든 헤드의 출력을 결합하여 d_model차원으로 변환
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries,keys, values, attn_mask, tau = None, delta = None):
        # B: 배치크기, L: 쿼리 길이 ( 쿼리토큰 개수), S: 키 길이 (키토큰 개수) , H 헤드수 
        # 각 변수 선언하는 부분
        B, L, _ = queries.shape,
        _, S, _ = keys.shape
        H = self.n_heads

        # VIEW -> 차원 모양 변경시 사용 : -1 넣을 경우 나머지 차원 자동 계산
        queries = self.query_projection(queries).view(B,L,H,-1)
        keys = self.key_projeciton(keys).view(B,S,H,-1)
        values = self.value_projection(values).view(B,S,H,-1)

        #실제 attention 계산 
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta = delta
        )
        out = out.view(B,L,-1)

        return self.out_projection(out), attn




