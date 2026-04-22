import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# 这里主要就是分为，
  # 准备模型和tokenizer，输入messages，
  # tokenizer编码处理，transformer推理输出，tokenizer解码输出
# 这里是真实的引入了一个0.5b的Qwen模型
# 加载模型和 Tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

# 第1步：构造输入（Tokenizer 编码）
messages = [
    {"role": "system", "content": "请提取用户问题中的学校、专业和意图，输出JSON"},
    {"role": "user", "content": "清华计算机保研需要什么条件"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("=== 第1步：Tokenizer 编码 ===")
print(f"输入 token 数量: {inputs['input_ids'].shape[1]}")

# 第2步：模型推理（Transformer 计算）
print("\n=== 第2步：Transformer 推理 ===")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

print(f"输出 token 数量: {outputs.shape[1]}")
print(f"其中新生成的 token: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

# 第3步：解码输出（Tokenizer 解码）
new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n=== 第3步：Tokenizer 解码 ===")
print(f"模型回答: {response}")