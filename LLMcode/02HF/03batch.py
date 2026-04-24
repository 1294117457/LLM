from transformers import AutoTokenizer
# 对tokenIDs进行batch
# 保证训练时多个数据统一格式
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = [
    "保研条件",
    "清华大学计算机系的保研要求和具体流程是什么",
    "北大",
]

# 不做 padding — 每条长度不同，无法组成 batch
for t in texts:
    ids = tokenizer.encode(t)
    print(f"[{len(ids):>2} tokens] {t}")

print("\n--- 加 padding 和 truncation ---")

# 做 padding + truncation — 统一长度
# 批处理（Batch Processing）。
# 高并发服务器端：假设有 3 个不同的用户在同一毫秒内向服务器发起了提问。
#   为了压榨 GPU 的算力，服务器不会一个一个算，而是把这 3 个人的问题打包成一个 Batch（矩阵），让 GPU 一次性同时算出 3 个答案。
# 批量数据处理/模型微调：在微调模型时，我们习惯一下子把包含成百上千个句子的数据集分批喂给模型。
batch = tokenizer(
    texts,
    padding=True,          # 短序列用 pad_token 补齐到 batch 内最长
    truncation=True,       # 超过 max_length 的截断
    max_length=20,         # 最大长度
    return_tensors="pt",   # 返回 PyTorch Tensor
)

print(f"input_ids shape: {batch['input_ids'].shape}")      # (3, max_len)
print(f"attention_mask shape: {batch['attention_mask'].shape}")

# 查看 padding 效果
for i, t in enumerate(texts):
    ids = batch['input_ids'][i].tolist()
    mask = batch['attention_mask'][i].tolist()
    print(f"\n文本: {t}")
    print(f"  IDs:  {ids}")
    print(f"  Mask: {mask}")  # 1=真实token, 0=padding