class MLA(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2)
    核心思想: Q 低秩压缩 + KV 低秩压缩 + RoPE 解耦
    """
    def __init__(self, d, n_heads, d_nope, d_rope, d_v, d_q, d_kv):
        # d:      hidden_size (e.g. 5120)
        # d_nope: QK 不做 RoPE 的维度 (e.g. 128)
        # d_rope: QK 做 RoPE 的维度 (e.g. 64)
        # d_v:    V 的 head_dim (e.g. 128)
        # d_q:    Q 压缩维度 q_lora_rank (e.g. 1536)
        # d_kv:   KV 压缩维度 kv_lora_rank (e.g. 512)

        # === Q 路径: 低秩分解 ===
        self.W_DQ = nn.Linear(d, d_q, bias=False)       # 压缩
        self.q_norm = RMSNorm(d_q)
        self.W_UQ = nn.Linear(d_q, n_heads * (d_nope + d_rope), bias=False)  # 展开

        # === KV 路径: 低秩压缩 (KV 共享压缩) ===
        # 输出 = compressed_KV [d_kv] + shared RoPE key [d_rope]
        self.W_DKV = nn.Linear(d, d_kv + d_rope, bias=False)
        self.kv_norm = RMSNorm(d_kv)
        # 从压缩 latent 展开 K_nope 和 V
        self.W_UKV = nn.Linear(d_kv, n_heads * (d_nope + d_v), bias=False)

        # === RoPE: 只对 rope 部分做 ===
        self.rope = RotaryEmbedding(d_rope)

        # === 输出 ===
        self.W_O = nn.Linear(n_heads * d_v, d, bias=False)

    def forward(self, h, positions):
        # h: [batch, seq, d]

        # Step 1: Q 低秩压缩 + 展开
        q_c = self.W_DQ(h)                          # [B,S,d_q]
        q_c = self.q_norm(q_c)
        q = self.W_UQ(q_c)                          # [B,S, n_heads*(d_nope+d_rope)]
        q = q.view(B, S, n_heads, d_nope + d_rope)
        q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)

        # Step 2: KV 低秩压缩 c_kv一个向量同时包含KV的共享压缩表示 K^C 和 后 d_rope 维K^R 位置分量
        c_kv = self.W_DKV(h)                        # [B,S, d_kv + d_rope]
        kv_c, k_rope = c_kv.split([d_kv, d_rope], dim=-1)

        # Step 3: KV 展开 
        kv_c = self.kv_norm(kv_c)
        kv = self.W_UKV(kv_c)                       # [B,S, n_heads*(d_nope+d_v)]
        kv = kv.view(B, S, n_heads, d_nope + d_v)
        k_nope, v = kv.split([d_nope, d_v], dim=-1)

        # Step 4: RoPE (仅对 rope 部分)
        q_rope, k_rope = self.rope(positions, q_rope, k_rope)

        # Step 5: 拼接完整 Q, K
        q = torch.cat([q_nope, q_rope], dim=-1)     # [B,S, n_heads, d_nope+d_rope]
        k = torch.cat([k_nope, k_rope], dim=-1)     # 同上

        # Step 6: 标准 scaled dot-product attention
        # (生产环境用 FlashAttention, 面试可以用这个)
        scale = (d_nope + d_rope) ** -0.5
        attn = F.softmax(q @ k.transpose(-2,-1) * scale, dim=-1)
        o = attn @ v                                 # [B,S, n_heads, d_v]

        # Step 7: 输出投影
        return self.W_O(o.flatten(-2))