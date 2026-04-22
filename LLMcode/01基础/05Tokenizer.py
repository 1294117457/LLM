# 编码：文字 → 数字（token IDs）→ 送入模型
# 解码：模型输出数字 → 转回文字 → 返回给用户

# 为什么不直接一个字一个数字？因为效率问题：颗粒度低了效率低，颗粒度高了词汇量会非常大

# # 字级别切分："清" "华" "大" "学" → 4 个 token
#             词汇表需要几千个汉字，但无法处理没见过的新词

# 词级别切分："清华大学" → 1 个 token
#             词汇表会非常巨大，新词（如网络用语）无法处理

# BPE 切分（实际使用）："清华" "大学" → 2 个 token
#             高频的字组合并成一个 token，低频的拆开
#             平衡了效率和覆盖度

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

text = "清华大学计算机系的保研要求是什么"

# ========== 编码：文字 → token IDs ==========
token_ids = tokenizer.encode(text)
print(f"原文: {text}")
print(f"Token IDs: {token_ids}")
print(f"Token 数量: {len(token_ids)}")

# ========== 查看每个 token 对应的文字 ==========
tokens = tokenizer.tokenize(text)
print(f"Token 切分: {tokens}")

# ========== 解码：token IDs → 文字 ==========
decoded = tokenizer.decode(token_ids)
print(f"解码还原: {decoded}")