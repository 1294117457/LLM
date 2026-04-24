from datasets import load_dataset

dataset = load_dataset("json", data_files="train_data.json", split="train")
print(dataset.num_rows)
# 按比例划分
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"训练集: {len(split_dataset['train'])} 条")
print(f"验证集: {len(split_dataset['test'])} 条")
print(split_dataset)
# 也可以用多个文件
# dataset = load_dataset("json", data_files={"train": "train.json", "test": "val.json"})