from transformers import pipeline

# # 文本生成
# generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
# result = generator("请问清华大学计算机保研需要什么条件？", max_new_tokens=100)
# print(result[0]["generated_text"])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 加载 Tokenizer 和模型是两个独立操作
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # 半精度，省显存
    device_map="auto",           # 自动分配到 GPU
)

# 构造输入
messages = [
    {"role": "system", "content": "你是一个语句提纯助手"},
    {"role": "user", "content": "清华计算机保研需要什么条件"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)