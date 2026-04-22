from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print("\n--- 1. Tokenizer 基础配置信息 ---\n",tokenizer)

# 特殊 token
# print(f"BOS (开始): {tokenizer.bos_token} → ID: {tokenizer.bos_token_id}")
# print(f"EOS (结束): {tokenizer.eos_token} → ID: {tokenizer.eos_token_id}")
# print(f"PAD (填充): {tokenizer.pad_token} → ID: {tokenizer.pad_token_id}")

# Chat 模板 — 这是微调时最重要的部分
messages = [
    {"role": "system", "content": "你是一个语句提纯助手"},
    {"role": "user", "content": "清华计算机保研需要什么条件"},
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print("\nChat 模板格式化结果：")
print(formatted)
print("\n--- 2. Tokenizer 基础配置信息 ---\n",tokenizer)
