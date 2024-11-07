class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys = None, d_values = None):
        super(AttentionLayer,self).__init__():

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
        values = self.value_projection(B,S,H, -1)

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

