from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("json", data_files="train_data.json", split="train")

def format_to_chat(example):
    """将 Alpaca 格式转为 Chat Template 格式"""
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

dataset = dataset.map(format_to_chat)

print("格式化后的第一条数据：")
print(dataset[0]["text"])