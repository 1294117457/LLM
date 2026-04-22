import torch
import torch.nn.functional as F

# 模拟一个 4 个 token 的句子，每个 token 用 8 维向量表示
seq_len = 4
d_model = 8

tokens = torch.randn(seq_len, d_model)
token_names = ["清华", "计算机", "保研", "要求"]

# 模拟 Q K V 权重矩阵
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)

Q = tokens @ W_Q
K = tokens @ W_K

# 计算注意力分数
scores = Q @ K.T / (d_model ** 0.5)
attention = F.softmax(scores, dim=-1)

# 打印注意力矩阵
print("Attention 矩阵（每行表示一个 token 对其他 token 的关注度）：")
print(f"{'':>8}", end="")
for name in token_names:
    print(f"{name:>8}", end="")
print()

for i, name in enumerate(token_names):
    print(f"{name:>8}", end="")
    for j in range(seq_len):
        print(f"{attention[i][j].item():>8.3f}", end="")
    print()


    #           清华     计算机      保研      要求
    #   清华   1.000   0.000   0.000   0.000
    #  计算机   1.000   0.000   0.000   0.000
    #   保研   0.023   0.164   0.046   0.768
    #   要求   0.999   0.000   0.000   0.001
 

#  输入 token IDs
#       │
#       ▼
# ┌──────────────┐
# │ Token Embedding │  把 ID 转成向量（查表操作）
# │ + 位置编码      │  加上位置信息（让模型知道词序）
# └──────┬───────┘
#        │
#        ▼
# ┌──────────────────────────┐
# │ Transformer Block × N 层  │  ← N 通常是 6~96 层不等
# │                          │
# │  ┌─────────────────┐     │
# │  │ Multi-Head       │     │
# │  │ Self-Attention    │     │  每个 token 关注所有 token
# │  └────────┬────────┘     │
# │           │              │
# │  ┌────────▼────────┐     │
# │  │ Feed Forward     │     │  两层全连接网络，做非线性变换
# │  │ Network (FFN)    │     │
# │  └────────┬────────┘     │
# │           │              │
# └───────────┼──────────────┘
#             │  （重复 N 次）
#             ▼
#     最终每个 token 的表示向量
#             │
#             ▼
#     线性层 → Softmax → 预测下一个 token 的概率