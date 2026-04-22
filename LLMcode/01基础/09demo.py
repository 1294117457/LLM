import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

messages = [{"role": "user", "content": "1+1等于几？"}]
# 转为带特殊标记的拼接文本
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 这里分割并转为TokenIDs
input_ids = tokenizer(text, return_tensors="pt").to(model.device)["input_ids"]

print("逐 token 生成过程：")
print("-" * 50)

generated = input_ids
for step in range(20):
    with torch.no_grad():
        output = model(generated)

    # output.logits 是模型对下一个 token 的概率预测
    next_token_logits = output.logits[:, -1, :]  # 取最后一个位置的预测

    # 取概率最大的 token
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = probs.argmax(dim=-1, keepdim=True)
    confidence = probs.max().item()

    # 解码这个 token
    token_text = tokenizer.decode(next_token_id[0])

    # 检查是否生成了结束标记
    if next_token_id.item() == tokenizer.eos_token_id:
        print(f"Step {step+1}: [EOS] (结束生成)")
        break

    print(f"Step {step+1}: '{token_text}' (置信度: {confidence:.3f})")

    # 把新 token 拼到序列后面，继续生成
    generated = torch.cat([generated, next_token_id], dim=-1)

print("-" * 50)
full_response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"完整回答: {full_response}")