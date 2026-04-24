from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,                        # LoRA 秩：越大拟合能力越强，但参数也越多
    lora_alpha=32,               # 缩放系数，通常设为 2*r
    target_modules=[             # 在哪些层加 LoRA
        "q_proj", "k_proj",      # Attention 的 Q K 投影
        "v_proj", "o_proj",      # Attention 的 V 和输出投影
    ],
    lora_dropout=0.05,           # Dropout 比例
    bias="none",                 # 不训练 bias
    task_type=TaskType.CAUSAL_LM,
)

# 包装模型
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# 输出类似：trainable params: 1,048,576 || all params: 494,032,896 || trainable%: 0.2123

# Tokenizer (分词器)：系统最外层，将人类文本切块并转化为数字 ID 序列。
# Embedding 层 (词嵌入字典)：将数字 ID 查表，映射成包含语义的高维浮点数特征向量。
# 位置编码 (Positional Encoding, 如 RoPE)：因为矩阵计算没有先后顺序，该结构为特征向量注入一种数学上的“角度/位置”信息，让模型感知词的先后顺序。
# 自注意力机制 (Self-Attention 模块)：大模型的大脑，利用 Q、K、V 权重捕捉上下文中所有词的关联程度和信息融合。
# 激活函数 (Activation Function, 如 SiLU)：夹在各个线性层中间的一道数学公式，负责引入“非线性”（如剔除负数）。如果没有它，神经网络无论叠加多少层，本质上依然只是一条直线，无法拟合复杂的逻辑。
# 归一化层 (Normalization, 如 RMSNorm)：在每一层计算完后，将张量的数值强制缩放回一个稳定的区间，防止经过上百层计算后数值爆炸（导致算不出结果）或梯度消失。
# LM Head (语言模型预测头)：系统的最后一关。一个巨大的分类器矩阵，将最后一层输出的高维向量，映射成词汇表中数十万个候选字的概率得分（Logits）。