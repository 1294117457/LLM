from datasets import load_dataset

dataset = load_dataset("json", data_files="train_data.json", split="train")

# === map：对每条数据做变换（最常用） ===
def add_text_length(example):
    example["input_length"] = len(example["input"])
    return example

dataset = dataset.map(add_text_length)
print(dataset[0])  # 多了 input_length 字段

# === filter：筛选数据 ===
long_samples = dataset.filter(lambda x: x["input_length"] > 15)
print(f"长文本样本数: {len(long_samples)}")

# === select：按索引选取 ===
subset = dataset.select(range(2))
print(f"子集样本数: {len(subset)}")