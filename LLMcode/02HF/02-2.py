import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. 分别加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2. 构造输入（Instruct 类型的模型需要使用对话模板格式化）
messages = [{"role": "user", "content": "请问清华大学计算机保研需要什么条件？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 转换为模型需要的 Tensor 并放到 GPU 上
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print('\ntokenizer -- inputs:',inputs)

# 3. 关闭梯度计算，进行推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print('\nmodel -- outputs:',outputs)
# 4. 截取新生成的 Token IDs 并解码回文本
input_len = inputs["input_ids"].shape[1]
result_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

# print(result_text)

