from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learn
from utils.metrics import metric
import torch 
import torch.nn as nn 
from torch import optim
import os
import time
import warnings
import numpy as np 

#발생하는 경고 메세지 꺼버리기
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self,args):
        # 자식의 init 통해서 args 받고 부모 클래스로 args 전달해 줘야함
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        return model 
    
    def _get_data(self,flag):
        data_set, data_loader = data_provider(self.args, flag)