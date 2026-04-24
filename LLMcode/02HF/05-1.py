from datasets import load_dataset

# 加载本地 JSON
  # "json"：告诉 Hugging Face 的 datasets 库，你要使用 JSON 解析器来读取文件。
  # data_files="train_data.json"：指定引用的本地文件路径（也可以是一个列表，传入多个文件）。
  # split="train"：大模型数据集通常分为训练集（train）、验证集（validation）和测试集（test）。
  # 由于本地加载时没有区分，系统默认会把文件里的所有数据归入名为 "train" 的切片中。
  # 加上这个参数，函数会直接返回一个可迭代的 Dataset 对象，如果不加，返回的会是一个包含 train 键的字典 DatasetDict。
dataset = load_dataset("json", data_files="train_data.json", split="train")
print(dataset)
print(dataset[0])
print(f"列名: {dataset.column_names}")
print(f"样本数: {len(dataset)}")