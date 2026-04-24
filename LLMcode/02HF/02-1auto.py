from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 打印模型结构（可以看到每一层）
# 1. 基础词嵌入层Embedding (embed_tokens): Embedding(151936, 896)
#   巨型查表字典。
#   把你之前看到的数字 ID（如 151644）转换成含有数学意义的向量。
#   参数意思：151936 是词表总词数（Qwen认识这么多子词），
#   896 是将每个词转换成 896 维的特征向量。
# 2. 核心大脑层 (Transformer Blocks)  (layers): ModuleList( (0-23): 24 x Qwen2DecoderLayer )
#   模型真正的“思考”中枢。
#   有 24 层一模一样的处理模块叠在一起。
#   每一层都包含以下两大部分：
#     A. 自注意力机制 (Qwen2Attention)
#       q_proj, k_proj, v_proj：就是你之前学过的计算 Q、K、V 的权重矩阵。
#         注意：q_proj 输出是 896 维，但 k_proj 和 v_proj 降到了 128 维。这叫分组查询注意力 (GQA/MQA)，是大模型为了省显存、提速做的高级优化。
#       o_proj：将多头注意力计算完的结果重新融合，变回 896 维，交给下一步。
#     B. 前馈神经网络 (Qwen2MLP) (对应你之前问过的 FFN)
#        作用：Attention 负责查上下文关联，
#        MLP 负责做特征的非线性变换和知识提取（存大模型的死记硬背记忆）。
#        gate_proj, up_proj, down_proj：这也是大模型的高级优化（SwiGLU 结构）。它先用两个矩阵把 896 维暴增膨胀到 4864 维（升维思考），经过 SiLU 激活函数，最后用 down_proj 压缩回 896 维输出。
#     C. 归一化层 (RMSNorm)
#       input_layernorm 等的作用是防止一层层计算下来数值爆炸或消失，
#       保证数据在一个稳定的区间内。这也是一种省显存的改良版归一化。
# 3. 位置编码 (rotary_emb): Qwen2RotaryEmbedding()
#   作用：旋转位置编码（RoPE）。
#   模型本身不知道哪个词在前哪个词在后，这个组件给每个 896 维向量注入一种“旋转角度”，让模型能精确感知词距（比如“清华”在“保研”前几个位置）。
# 4. 输出预测头 (LM Head) (lm_head): Linear(in_features=896, out_features=151936, bias=False)
#   作用：最后一关的“投票器”。
#   经过 24 层的思考后，最终输出一个极其精华的 896 维向量。
#   lm_head 负责把这个向量映射回那 151936 个候选词上，得出每一个词的得分（Logits）。
#   得分最高的，就是模型要开口说的下一个字

# print(model)


# # 查看参数量
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"总参数: {total_params / 1e6:.1f}M")
# print(f"可训练参数: {trainable_params / 1e6:.1f}M")

from transformers import pipeline
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
generator = pipeline("text-generation", model=model_name, device_map="auto")
result = generator("请问清华大学计算机保研需要什么条件？", max_new_tokens=100)
print(result[0]["generated_text"])
print(result)