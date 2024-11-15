import os
import torch
from models import iTransformer


class Exp_basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {"iTransformer": iTransformer}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # 이 클래스를 상속받는 곳에서 아래 함수 오버라이딩을 강제하기 위하여 에러레이즈

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))

        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
