from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "system", "content": "你是一个语句提纯助手，将用户问句转为结构化JSON"},
    {"role": "user", "content": "清华计算机保研需要什么条件"},
    {"role": "assistant", "content": '{"school":"清华大学","major":"计算机","intent":"保研条件"}'},
]

# tokenize=False → 返回格式化后的纯文本（看格式用）
formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
print("格式化文本：")
print(formatted_text)
print()

# tokenize=True → 直接返回 token IDs（训练时用）
token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
print(f"Token IDs 数量: {len(token_ids)}")
print(formatted_text)

# add_generation_prompt=True → 推理时用，在末尾加上助手回复的开头标记
inference_text = tokenizer.apply_chat_template(
    messages[:2],  # 只有 system + user
    tokenize=False,
    add_generation_prompt=True,
)
print("\n推理时的输入（带 generation prompt）：")
print(inference_text)