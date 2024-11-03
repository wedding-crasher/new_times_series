import torch 
import torch.nn as nn
import numpy as np 


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type = 'fixed', freq= 'h', dropout = 0.1):
        super(DataEmbedding_inverted, self).__init__()
        #Linear는 마지막 차원에만 영향을 미친다. 
        self.value_embedding = nn.Linear(c_in,d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x,x_mark):
        x = x.permute(0,2,1)
        #Input시 (배치, Time, Variate) -> (배치, Variate, Time)
        if x_mark is None: 
            #Univariate 기준
            x = self.value_embedding(x)
        
        else: 
            # Multivariate일 경우 -> Variate dimension 기준으로 이어붙이기 
            # torch.cat(a,b): a-> 이어붙일 것들 리스트로, 1(붙일 차원)
            x = self.value_embedding(torch.cat([x,x_mark.permute(0,2,1)],1))           
        # x: [Batch, Variate, d_model]
        return self.dropout(x)
    
    
        