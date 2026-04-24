from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# tokenizer.tokenize(text)：
    # 负责**“切词”**。
    # 它将一整句话切分成大模型词表里存在的最小基本单位。输出的是一个字符串列表。
text = "清华大学计算机系的保研要求"
ids = tokenizer.encode(text)
tokens = tokenizer.tokenize(text)

print("逐 token 映射：")
for token, tid in zip(tokens, ids):
    decoded = tokenizer.decode([tid])
    print(f"  token: {token:>10}  →  ID: {tid:>8}  →  decode back: '{decoded}'")