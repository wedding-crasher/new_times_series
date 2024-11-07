import torch 



class TriangularCausalMask():
    # B: 배치사이즈, L: 시퀀스 길이(마스크 적용 시퀀스 길이), device: 마스크 저장장치 
    def __init__(self, B, L, device = "cpu"):
        # 트랜스포머 모델에서 각각의 쿼리_키 쌍에 대해 배치와 시퀀스 길이에 맞게 만들어야 하므로, 4차원 으로 생성 
        mask_shape = [B,1,L,L]
        # 텐서 생성시 그래디언트 계산을 비활성화
        with torch.no_grad():
            
            #torch.ones로 4차원 텐서 만들고 bool로 채운다, triu(diagonal=1)은 텐서의 상삼각 부분을 유지하고 나머지를 False로 설정. 
            #이 마스크는 diagonal =1 옵션때문에 자신과 그 이전의 토큰 만 봄. False는 참고 가능, True는 마스킹된거임. 
            self._mask = torch.triu(torch.ones(mask_shape, dtype = torch.bool), diagonal = 1).to(device)
        
        @property
        def mask(self):
            return self._mask