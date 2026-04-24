# Hugging Face 生态完全学习指南

> 前置：已掌握 PyTorch 基础（Tensor、自动求导、训练循环）、Transformer 架构（Self-Attention、Q-K-V）、Tokenizer（BPE、Chat Template）
> 目标：7 天系统掌握 Hugging Face 全套工具链，为 QLoRA 微调语句提纯模型做好准备
> 环境：Google Colab（免费 T4 GPU）或本地 Python + GPU

---

## 总览：7 天学习路线

```
Day 1  transformers 核心 API — 模型加载、推理、Pipeline
Day 2  Tokenizer 深度使用 — 编码细节、padding、truncation、batch 处理
Day 3  datasets 库 — 加载、处理、格式化训练数据
Day 4  Trainer API — 用 Hugging Face 原生训练器跑通一个文本分类微调
Day 5  PEFT / LoRA — 参数高效微调原理与实战
Day 6  TRL + SFTTrainer — 指令微调（SFT）实战，跑通你的语句提纯场景
Day 7  模型导出与部署 — 合并权重、量化导出、Ollama 本地部署
```

---

## Day 1：transformers 核心 API

### 目标
- 理解 `AutoModel`、`AutoTokenizer`、`AutoModelForCausalLM` 的加载机制
- 学会用 `pipeline` 做零代码推理
- 理解模型的输入输出数据结构

### 1.1 环境安装

```bash
pip install transformers torch accelerate
```

### 1.2 Pipeline — 最快的推理方式

Pipeline 是 Hugging Face 封装的"一行代码做推理"工具，自动处理 Tokenizer + 模型 + 后处理。

```python
from transformers import pipeline

# 文本生成
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
result = generator("请问清华大学计算机保研需要什么条件？", max_new_tokens=100)
print(result[0]["generated_text"])
```

Pipeline 支持的任务类型（了解即可，重点用 text-generation）：

| 任务 | pipeline 名称 | 说明 |
|------|--------------|------|
| 文本生成 | `text-generation` | GPT/Qwen 等生成模型 |
| 文本分类 | `text-classification` | 情感分析、主题分类 |
| Token 分类 | `token-classification` | NER 命名实体识别 |
| 问答 | `question-answering` | 抽取式问答 |
| 翻译 | `translation` | 机器翻译 |
| 摘要 | `summarization` | 文本摘要 |

### 1.3 手动加载模型 — 理解 Pipeline 背后发生了什么

```python
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
```

**理解要点**：
- `AutoTokenizer.from_pretrained()` — 根据模型名自动下载并加载正确的 Tokenizer
- `AutoModelForCausalLM.from_pretrained()` — 加载因果语言模型（GPT/Qwen 这类自回归生成模型）
- `torch_dtype=torch.float16` — 用半精度加载，显存占用减半
- `device_map="auto"` — 自动把模型放到可用的 GPU 上
- `model.generate()` — 封装了逐 token 生成的循环（你在第 4 篇文档里手写过的那个）

### 1.4 Auto 类家族

Hugging Face 用 `Auto` 前缀封装了"根据模型名自动选择正确类"的逻辑：

| Auto 类 | 用途 | 何时使用 |
|---------|------|---------|
| `AutoTokenizer` | 加载 Tokenizer | 几乎所有场景 |
| `AutoModelForCausalLM` | 加载生成模型 | GPT/Qwen/LLaMA 做文本生成 |
| `AutoModelForSequenceClassification` | 加载分类模型 | 文本分类任务 |
| `AutoModel` | 加载基础模型（不含任务头） | 取 Embedding、做特征提取 |
| `AutoConfig` | 加载模型配置 | 查看/修改模型超参数 |

### 1.5 查看模型结构

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 打印模型结构（可以看到每一层）
print(model)

# 查看参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total_params / 1e6:.1f}M")
print(f"可训练参数: {trainable_params / 1e6:.1f}M")
```

### 1.6 generate() 参数详解

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,     # 最多生成 100 个新 token
    temperature=0.7,        # 温度：越低越确定，越高越随机
    top_p=0.9,              # nucleus sampling：只从累积概率前 90% 的 token 中采样
    top_k=50,               # 只从概率最大的 50 个 token 中采样
    do_sample=True,         # 是否采样（False = 贪心，永远选概率最大的）
    repetition_penalty=1.1, # 重复惩罚，防止模型重复输出
)
```

| 参数 | 影响 | 推荐值 |
|------|------|--------|
| `temperature` | 控制随机性，0.0=确定，1.0=正常随机 | 0.1-0.3（提纯任务需要确定性输出） |
| `top_p` | 截断低概率 token | 0.9 |
| `do_sample` | 是否随机采样 | False（提纯任务用贪心更稳定） |
| `max_new_tokens` | 最大生成长度 | 根据你的 JSON 输出长度设置 |

### Day 1 练习

1. 用 `pipeline` 跑通一个文本生成任务
2. 用手动加载方式（AutoTokenizer + AutoModelForCausalLM）实现同样的功能
3. 调整 `temperature` 为 0.1、0.7、1.5，观察输出差异
4. 打印模型结构，找到 `self_attn`（Self-Attention 层）和 `mlp`（FFN 层）

---

## Day 2：Tokenizer 深度使用

### 目标
- 掌握 `__call__`、`encode`、`decode`、`batch_encode_plus` 的区别
- 理解 padding、truncation、attention_mask
- 学会处理批量数据

### 2.1 Tokenizer 调用方式对比

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
text = "清华大学计算机系保研条件"

# 方式1：encode — 只返回 token IDs（列表）
ids = tokenizer.encode(text)
print(f"encode: {ids}")  # [101592, 103775, ...]

# 方式2：tokenize — 只返回 token 字符串（不含 ID）
tokens = tokenizer.tokenize(text)
print(f"tokenize: {tokens}")  # ['清华大学', '计算机', '系', ...]

# 方式3：__call__ — 返回完整字典（模型需要的所有输入）
encoded = tokenizer(text, return_tensors="pt")
print(f"__call__: {encoded.keys()}")  # dict_keys(['input_ids', 'attention_mask'])
print(f"  input_ids shape: {encoded['input_ids'].shape}")
print(f"  attention_mask shape: {encoded['attention_mask'].shape}")
```

**核心区别**：
- `encode()` → 最简单，返回 ID 列表，适合调试
- `tokenize()` → 返回 token 文本，适合可视化
- `tokenizer()` / `__call__()` → 返回完整输入字典（`input_ids` + `attention_mask`），**训练和推理时必须用这个**

### 2.2 Padding 和 Truncation

训练时每个 batch 里的序列长度必须一致，需要 padding（短的补齐）和 truncation（长的截断）。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = [
    "保研条件",
    "清华大学计算机系的保研要求和具体流程是什么",
    "北大",
]

# 不做 padding — 每条长度不同，无法组成 batch
for t in texts:
    ids = tokenizer.encode(t)
    print(f"[{len(ids):>2} tokens] {t}")

print("\n--- 加 padding 和 truncation ---")

# 做 padding + truncation — 统一长度
batch = tokenizer(
    texts,
    padding=True,          # 短序列用 pad_token 补齐到 batch 内最长
    truncation=True,       # 超过 max_length 的截断
    max_length=20,         # 最大长度
    return_tensors="pt",   # 返回 PyTorch Tensor
)

print(f"input_ids shape: {batch['input_ids'].shape}")      # (3, max_len)
print(f"attention_mask shape: {batch['attention_mask'].shape}")

# 查看 padding 效果
for i, t in enumerate(texts):
    ids = batch['input_ids'][i].tolist()
    mask = batch['attention_mask'][i].tolist()
    print(f"\n文本: {t}")
    print(f"  IDs:  {ids}")
    print(f"  Mask: {mask}")  # 1=真实token, 0=padding
```

**理解要点**：
- `attention_mask`：告诉模型哪些位置是真实 token（1），哪些是 padding（0）
- 模型在计算 Attention 时会忽略 mask=0 的位置
- `padding=True` 补齐到 batch 内最长序列
- `padding="max_length"` 补齐到指定的 `max_length`

### 2.3 Chat Template 深入

这是微调时最关键的部分——必须确保训练数据和推理时使用相同的格式。

```python
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

# add_generation_prompt=True → 推理时用，在末尾加上助手回复的开头标记
inference_text = tokenizer.apply_chat_template(
    messages[:2],  # 只有 system + user
    tokenize=False,
    add_generation_prompt=True,
)
print("\n推理时的输入（带 generation prompt）：")
print(inference_text)
```

**核心理解**：
- 每个模型的 Chat Template 格式不同（Qwen 用 `<|im_start|><|im_end|>`，LLaMA 用 `[INST][/INST]`）
- 训练时包含完整对话（system + user + assistant），模型学习生成 assistant 部分
- 推理时只给 system + user + generation prompt，让模型接着生成

### 2.4 Token ID ↔ 文字的精确映射

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

text = "清华大学计算机系的保研要求"
ids = tokenizer.encode(text)
tokens = tokenizer.tokenize(text)

print("逐 token 映射：")
for token, tid in zip(tokens, ids):
    decoded = tokenizer.decode([tid])
    print(f"  token: {token:>10}  →  ID: {tid:>8}  →  decode back: '{decoded}'")
```

### Day 2 练习

1. 对 5 个不同长度的句子做 batch 编码，观察 padding 和 attention_mask
2. 用 `apply_chat_template` 格式化 3 组不同的对话，打印完整文本，熟悉 Qwen 的模板格式
3. 尝试对超长文本设置 `max_length=50` 并 `truncation=True`，观察截断效果
4. （进阶）自己手写一个函数，把 Alpaca 格式的 `{"instruction", "input", "output"}` 转成 messages 列表，再用 `apply_chat_template` 格式化

---

## Day 3：datasets 库 — 数据加载与处理

### 目标
- 学会用 `datasets` 加载本地 JSON/CSV 数据
- 掌握 `map`、`filter`、`select` 等数据处理方法
- 学会将数据格式化为模型训练所需的格式

### 3.1 安装与基本加载

```bash
pip install datasets
```

```python
from datasets import load_dataset

# 加载 Hugging Face Hub 上的公开数据集
dataset = load_dataset("yelp_review_full", split="train[:100]")
print(dataset)
print(dataset[0])
print(dataset.column_names)
```

### 3.2 加载本地数据（你的实际场景）

先创建一个示例训练数据文件：

```python
import json

train_data = [
    {
        "instruction": "请将以下用户问句提取为结构化意图JSON",
        "input": "我想了解清华大学计算机系的保研要求和流程",
        "output": '{"school":"清华大学","major":"计算机","intent":"保研要求+流程","keywords":["保研","要求","流程"]}'
    },
    {
        "instruction": "请将以下用户问句提取为结构化意图JSON",
        "input": "北大数学系保研需要GPA多少",
        "output": '{"school":"北京大学","major":"数学","intent":"GPA要求","keywords":["保研","GPA"]}'
    },
    {
        "instruction": "请将以下用户问句提取为结构化意图JSON",
        "input": "浙大软件工程考研分数线",
        "output": '{"school":"浙江大学","major":"软件工程","intent":"考研分数线","keywords":["考研","分数线"]}'
    },
]

with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
```

```python
from datasets import load_dataset

# 加载本地 JSON
dataset = load_dataset("json", data_files="train_data.json", split="train")
print(dataset)
print(dataset[0])
print(f"列名: {dataset.column_names}")
print(f"样本数: {len(dataset)}")
```

### 3.3 数据处理：map、filter、select

```python
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
```

### 3.4 关键操作：将数据格式化为训练格式

SFTTrainer 需要数据集中有一个文本字段（通常叫 `text`），内容是经过 Chat Template 格式化的完整对话。

```python
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
```

### 3.5 划分训练集和验证集

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="train_data.json", split="train")

# 按比例划分
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"训练集: {len(split_dataset['train'])} 条")
print(f"验证集: {len(split_dataset['test'])} 条")

# 也可以用多个文件
# dataset = load_dataset("json", data_files={"train": "train.json", "test": "val.json"})
```

### 3.6 数据集常用属性和方法速查

| 方法/属性 | 用途 |
|-----------|------|
| `dataset.column_names` | 查看所有列名 |
| `dataset.num_rows` / `len(dataset)` | 样本数 |
| `dataset.map(func)` | 对每条数据应用函数 |
| `dataset.filter(func)` | 筛选符合条件的数据 |
| `dataset.select(indices)` | 按索引选取 |
| `dataset.shuffle(seed=42)` | 随机打乱 |
| `dataset.train_test_split()` | 划分训练/验证集 |
| `dataset.to_pandas()` | 转为 Pandas DataFrame |
| `dataset.save_to_disk("path")` | 保存到本地 |

### Day 3 练习

1. 创建一个包含 10+ 条语句提纯数据的 JSON 文件
2. 用 `load_dataset` 加载，用 `map` 将 Alpaca 格式转为 Chat Template 格式
3. 划分训练集（90%）和验证集（10%）
4. 打印格式化后的 3 条数据，确认格式正确
5. （进阶）用 `filter` 筛选出 output 中包含特定学校的样本

---

## Day 4：Trainer API — 文本分类微调实战

### 目标
- 理解 Hugging Face `Trainer` 的训练流程
- 跑通一个完整的文本分类微调（在真实数据集上）
- 理解 `TrainingArguments` 中的关键超参数

> 为什么先做文本分类？因为它比生成任务简单，流程完全一样，适合先熟悉 Trainer API。

### 4.1 任务：情感分类微调

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

# 1. 加载数据集（IMDb 电影评论情感分类）
dataset = load_dataset("imdb")
# 取一个小子集快速实验
small_train = dataset["train"].shuffle(seed=42).select(range(1000))
small_test = dataset["test"].shuffle(seed=42).select(range(200))

# 2. 加载 Tokenizer 和模型
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

small_train = small_train.map(tokenize_function, batched=True)
small_test = small_test.map(tokenize_function, batched=True)

# 4. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="./imdb-finetune",     # 输出目录
    num_train_epochs=3,               # 训练轮数
    per_device_train_batch_size=16,   # 每个 GPU 的 batch size
    per_device_eval_batch_size=16,
    eval_strategy="epoch",            # 每个 epoch 结束时评估
    save_strategy="epoch",            # 每个 epoch 保存一次
    learning_rate=2e-5,               # 学习率
    weight_decay=0.01,                # 权重衰减（正则化）
    logging_steps=50,                 # 每 50 步打印一次 loss
    load_best_model_at_end=True,      # 训练结束后加载最好的 checkpoint
    report_to="none",                 # 不上报到 wandb（学习阶段先关闭）
)

# 6. 创建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_test,
    compute_metrics=compute_metrics,
)

trainer.train()

# 7. 评估
results = trainer.evaluate()
print(f"准确率: {results['eval_accuracy']:.4f}")
```

### 4.2 TrainingArguments 关键参数详解

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `num_train_epochs` | 遍历全量数据的次数 | 3-5（微调够用） |
| `per_device_train_batch_size` | 每 GPU 每步处理的样本数 | 4-16（显存不够就降） |
| `learning_rate` | 学习率 | 2e-5（分类）/ 2e-4（LoRA） |
| `weight_decay` | 权重衰减，防过拟合 | 0.01-0.1 |
| `warmup_steps` | 预热步数（开始时 lr 从 0 升到设定值） | 总步数的 5-10% |
| `logging_steps` | 每多少步打印一次 loss | 10-50 |
| `eval_strategy` | 评估频率 | "epoch" 或 "steps" |
| `fp16` / `bf16` | 混合精度训练（省显存加速） | True（有 GPU 就开） |
| `gradient_accumulation_steps` | 梯度累积步数（等效增大 batch size） | 显存不够时设 2-8 |

### 4.3 理解训练过程中的日志

```
{'loss': 0.6932, 'learning_rate': 1.9e-05, 'epoch': 0.16}
{'loss': 0.5124, 'learning_rate': 1.6e-05, 'epoch': 0.32}
{'loss': 0.3287, 'learning_rate': 1.3e-05, 'epoch': 0.48}
...
```

- `loss` 应该逐步下降（模型在学习）
- `learning_rate` 逐步衰减（warmup 之后的 decay）
- `epoch` 表示训练进度

如果 loss 不下降：学习率可能太小或太大
如果 loss 降到 0 附近：可能过拟合了（训练集背下来了但泛化不好）

### 4.4 保存和加载训练好的模型

```python
# 保存
trainer.save_model("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")

# 加载
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./my-finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")

# 使用
inputs = tokenizer("This movie is amazing!", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()
print(f"预测: {'正面' if prediction == 1 else '负面'}")
```

### Day 4 练习

1. 在 Colab 上运行完整的 IMDb 情感分类微调
2. 观察训练日志中 loss 的下降趋势
3. 尝试修改 `learning_rate`（试 1e-3 和 1e-6），观察对 loss 的影响
4. 保存模型，重新加载后进行推理，验证模型工作正常
5. （进阶）尝试将 `num_train_epochs` 改为 10，观察是否出现过拟合（训练 loss 很低但验证 accuracy 不再提升）

---

## Day 5：PEFT / LoRA — 参数高效微调

### 目标
- 理解为什么需要 LoRA（全量微调的显存问题）
- 理解 LoRA 的原理（低秩矩阵分解）
- 学会用 `peft` 库给模型加 LoRA
- 理解 QLoRA = 量化 + LoRA

### 5.1 为什么需要 LoRA

```
全量微调 Qwen2.5-0.5B：
  参数量 = 5 亿
  显存 = 模型参数(2GB) + 梯度(2GB) + 优化器状态(4GB) ≈ 8GB
  → 一张 3060 勉强能跑

全量微调 Qwen2.5-7B：
  参数量 = 70 亿
  显存 ≈ 70-80GB → 需要 A100 80GB 或多卡

LoRA 微调 Qwen2.5-7B：
  只训练 LoRA 参数 ≈ 原始参数的 0.1%-1%
  显存 ≈ 8-16GB → 一张 3060/3090 就能跑
```

### 5.2 LoRA 原理（直觉理解）

```
原始模型的权重矩阵 W（比如 4096 × 4096 = 1600万参数）

全量微调：直接修改 W 的每一个值 → 需要存储完整梯度

LoRA 的思路：
  W 的变化量 ΔW 可以用两个小矩阵相乘来近似
  ΔW = A × B
  其中 A 是 4096 × r，B 是 r × 4096（r 通常取 8 或 16）

  训练时：冻结原始 W，只训练 A 和 B
  参数量：4096 × 16 × 2 = 131,072（只有原来的 0.8%）

  推理时：W_new = W + A × B（合并后和原模型结构完全一样）
```

### 5.3 安装 peft

```bash
pip install peft
```

### 5.4 给模型加 LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,                        # LoRA 秩：越大拟合能力越强，但参数也越多
    lora_alpha=32,               # 缩放系数，通常设为 2*r
    target_modules=[             # 在哪些层加 LoRA
        "q_proj", "k_proj",      # Attention 的 Q K 投影
        "v_proj", "o_proj",      # Attention 的 V 和输出投影
    ],
    lora_dropout=0.05,           # Dropout 比例
    bias="none",                 # 不训练 bias
    task_type=TaskType.CAUSAL_LM,
)

# 包装模型
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# 输出类似：trainable params: 1,048,576 || all params: 494,032,896 || trainable%: 0.2123
```

### 5.5 LoRA 配置参数详解

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `r` | 秩（rank）：LoRA 矩阵的中间维度 | 8-32（任务简单用 8-16，复杂用 32-64） |
| `lora_alpha` | 缩放系数 | 一般设为 2×r |
| `target_modules` | 在哪些模块上加 LoRA | Attention 层的 q/k/v/o_proj |
| `lora_dropout` | Dropout | 0.05-0.1 |
| `bias` | 是否训练 bias | "none" |

**如何选择 target_modules**：
```python
# 查看模型有哪些线性层可以加 LoRA
from peft import get_peft_model
print(model)  # 打印模型结构，找到所有 Linear 层的名字
```

### 5.6 QLoRA — 量化 + LoRA

QLoRA 在 LoRA 的基础上，把原始模型量化到 4-bit，进一步降低显存：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",         # NormalFloat4 量化类型（效果最好）
    bnb_4bit_compute_dtype=torch.float16,  # 计算时用 float16
    bnb_4bit_use_double_quant=True,    # 二次量化，进一步压缩
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# 加 LoRA（和之前一样）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

> 注意：`bitsandbytes` 目前对 Windows 支持有限，建议在 Linux（Colab/AutoDL）上使用 QLoRA。

### 5.7 LoRA 权重的保存与合并

```python
# === 保存 LoRA 权重（只有几 MB） ===
model.save_pretrained("./my-lora-weights")

# === 加载 LoRA 权重 ===
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my-lora-weights")

# === 合并 LoRA 到基座模型（部署时用） ===
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

合并后的模型和原始模型结构完全一样，但参数已经包含了 LoRA 的修改，可以直接部署。

### Day 5 练习

1. 给 Qwen2.5-0.5B 加 LoRA，打印可训练参数量，和全量参数对比
2. 尝试不同的 `r` 值（8, 16, 32, 64），观察可训练参数量的变化
3. 尝试不同的 `target_modules`（只加 q_proj vs 加 q/k/v/o_proj），对比参数量
4. 在 Colab 上尝试 QLoRA 加载（4-bit 量化），对比显存占用
5. 保存 LoRA 权重，查看文件大小（对比完整模型的几 GB，LoRA 权重只有几 MB）

---

## Day 6：TRL + SFTTrainer — 指令微调实战

### 目标
- 理解 SFTTrainer 和 Trainer 的区别
- 用 SFTTrainer 跑通你的语句提纯微调
- 学会数据格式化、训练监控、验证评估

### 6.1 SFTTrainer 是什么

```
Trainer（Day 4 学的）：
  通用训练器，适合分类、回归等判别任务
  需要手动 tokenize 数据

SFTTrainer（TRL 库提供）：
  专门为指令微调（Supervised Fine-Tuning）设计
  自动处理 Chat Template 格式化
  自动处理 tokenization
  内置支持 LoRA/QLoRA
  → 你的语句提纯微调直接用这个
```

### 6.2 安装

```bash
pip install trl peft bitsandbytes accelerate
```

### 6.3 完整微调代码（语句提纯场景）

```python
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ========================
# 第1步：准备训练数据
# ========================

train_data = []
for i in range(50):
    samples = [
        {
            "conversations": [
                {"role": "system", "content": "请将以下用户问句提取为结构化意图JSON"},
                {"role": "user", "content": "清华大学计算机系的保研要求和流程"},
                {"role": "assistant", "content": '{"school":"清华大学","major":"计算机","intent":"保研要求+流程","keywords":["保研","要求","流程"]}'},
            ]
        },
        {
            "conversations": [
                {"role": "system", "content": "请将以下用户问句提取为结构化意图JSON"},
                {"role": "user", "content": "北大数学系保研需要GPA多少"},
                {"role": "assistant", "content": '{"school":"北京大学","major":"数学","intent":"GPA要求","keywords":["保研","GPA"]}'},
            ]
        },
        {
            "conversations": [
                {"role": "system", "content": "请将以下用户问句提取为结构化意图JSON"},
                {"role": "user", "content": "浙大软件工程考研分数线是多少"},
                {"role": "assistant", "content": '{"school":"浙江大学","major":"软件工程","intent":"考研分数线","keywords":["考研","分数线"]}'},
            ]
        },
    ]
    train_data.extend(samples)

with open("sft_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# ========================
# 第2步：加载模型和 Tokenizer
# ========================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# QLoRA 量化配置（在 Linux/Colab 上使用）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# ========================
# 第3步：配置 LoRA
# ========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# ========================
# 第4步：加载数据集
# ========================

dataset = load_dataset("json", data_files="sft_train.json", split="train")

# ========================
# 第5步：配置训练参数并训练
# ========================

training_args = SFTConfig(
    output_dir="./intent-purifier-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    # 等效 batch_size = 4 * 2 = 8
    learning_rate=2e-4,               # LoRA 用较大学习率
    lr_scheduler_type="cosine",       # 余弦退火学习率
    warmup_ratio=0.1,                 # 前 10% 步数做预热
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                        # 混合精度训练
    max_seq_length=512,               # 最大序列长度
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

# ========================
# 第6步：保存 LoRA 权重
# ========================

trainer.save_model("./intent-purifier-lora")
tokenizer.save_pretrained("./intent-purifier-lora")

print("训练完成！LoRA 权重已保存到 ./intent-purifier-lora")
```

### 6.4 数据格式说明

SFTTrainer 支持多种数据格式：

**格式一：conversations 格式（推荐）**
```json
{
    "conversations": [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "期望输出"}
    ]
}
```
SFTTrainer 会自动用模型的 Chat Template 格式化。

**格式二：text 字段（已格式化）**
```json
{
    "text": "<|im_start|>system\n系统提示<|im_end|>\n<|im_start|>user\n用户输入<|im_end|>\n<|im_start|>assistant\n期望输出<|im_end|>"
}
```
你自己预先用 `apply_chat_template` 格式化好。

### 6.5 训练后验证

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型 + LoRA 权重
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./intent-purifier-lora")

# 测试推理
test_queries = [
    "清华计算机保研需要什么条件",
    "北大物理系考研难不难",
    "复旦经济学院推免名额有多少",
]

for query in test_queries:
    messages = [
        {"role": "system", "content": "请将以下用户问句提取为结构化意图JSON"},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"输入: {query}")
    print(f"输出: {response}")
    print()
```

### 6.6 训练监控（可选但推荐）

用 Weights & Biases（wandb）实时查看训练曲线：

```bash
pip install wandb
wandb login  # 输入你的 API key（免费注册 wandb.ai）
```

```python
# 在 SFTConfig 中启用
training_args = SFTConfig(
    ...
    report_to="wandb",
    run_name="intent-purifier-v1",
)
```

或者用 TensorBoard：

```python
training_args = SFTConfig(
    ...
    report_to="tensorboard",
    logging_dir="./logs",
)
```

```bash
tensorboard --logdir ./logs
```

### Day 6 练习

1. 用上面的完整代码在 Colab 上跑通一次微调
2. 用 3 条测试数据验证模型输出是否为合理的 JSON
3. 尝试修改 `num_train_epochs`（1 vs 3 vs 5），对比输出质量
4. 查看 `./intent-purifier-lora` 目录里的文件，理解 LoRA 权重的存储结构
5. （进阶）自己写 20 条更丰富的训练数据（不同学校、不同意图），重新训练，观察效果提升

---

## Day 7：模型导出与部署

### 目标
- 学会合并 LoRA 权重到基座模型
- 学会用 GGUF 格式导出（Ollama 需要）
- 学会在 Ollama 中部署自定义模型
- 跑通完整链路：训练 → 导出 → 本地部署 → 推理

### 7.1 合并 LoRA 权重

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 加载基座模型（全精度）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # 合并操作在 CPU 上进行
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 LoRA
model = PeftModel.from_pretrained(base_model, "./intent-purifier-lora")

# 合并
merged_model = model.merge_and_unload()

# 保存合并后的完整模型
merged_model.save_pretrained("./intent-purifier-merged")
tokenizer.save_pretrained("./intent-purifier-merged")

print("合并完成！模型已保存到 ./intent-purifier-merged")
```

### 7.2 转换为 GGUF 格式（Ollama 使用）

Ollama 使用 llama.cpp 的 GGUF 格式。需要用 llama.cpp 的转换工具：

```bash
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 安装 Python 依赖
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# 转换（FP16）
python convert_hf_to_gguf.py ../intent-purifier-merged --outtype f16 --outfile intent-purifier-f16.gguf

# 量化到 Q4_K_M（推荐，体积小推理快）
./llama-quantize intent-purifier-f16.gguf intent-purifier-Q4_K_M.gguf Q4_K_M
```

量化格式对比：

| 格式 | 大小（0.5B模型） | 质量 | 推荐度 |
|------|-----------------|------|--------|
| F16 | ~1 GB | 最高 | 显存充足时 |
| Q8_0 | ~0.5 GB | 很高 | 平衡之选 |
| Q4_K_M | ~0.3 GB | 良好 | 推荐（体积小，质量损失小） |
| Q4_0 | ~0.3 GB | 可接受 | 极致压缩 |

### 7.3 Ollama 部署

创建 Modelfile：

```
FROM ./intent-purifier-Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM "请将以下用户问句提取为结构化意图JSON"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
```

```bash
# 创建 Ollama 模型
ollama create intent-purifier -f Modelfile

# 测试
ollama run intent-purifier "清华计算机保研需要什么条件"
```

### 7.4 Python 调用本地 Ollama 模型

```python
import requests
import json

def purify_query(query: str) -> dict:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "intent-purifier",
            "prompt": query,
            "stream": False,
        },
    )
    result = response.json()["response"]
    return json.loads(result)

# 测试
test_queries = [
    "清华计算机保研需要什么条件",
    "北大数学系GPA要求多少",
    "浙大软件工程考研分数线",
]

for q in test_queries:
    try:
        result = purify_query(q)
        print(f"输入: {q}")
        print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print()
    except Exception as e:
        print(f"输入: {q} → 解析失败: {e}")
```

### 7.5 集成到 LangGraph（预览）

```python
from langchain_community.llms import Ollama

llm = Ollama(model="intent-purifier")

def purify_node(state):
    """LangGraph 中的语句提纯节点"""
    raw_query = state["user_query"]
    response = llm.invoke(raw_query)
    state["purified_intent"] = response
    return state
```

### Day 7 练习

1. 合并 LoRA 权重，保存完整模型
2. 转换为 GGUF 格式（在 Colab 或 Linux 上操作）
3. 用 Ollama 加载并测试 5 条查询
4. 用 Python requests 调用 Ollama API，验证端到端流程
5. （进阶）对比 Q4_K_M 和 F16 两种格式的输出质量差异

---

## 附录 A：常见问题排查

### A.1 显存不足（CUDA Out of Memory）

```python
# 方案1：降低 batch_size
per_device_train_batch_size=2  # 从 4 降到 2

# 方案2：开启梯度累积（等效 batch_size 不变，但每步显存更少）
gradient_accumulation_steps=4  # 等效 batch_size = 2 × 4 = 8

# 方案3：降低 max_seq_length
max_seq_length=256  # 从 512 降到 256

# 方案4：使用 QLoRA（4-bit 量化）
load_in_4bit=True
```

### A.2 Loss 不下降

```
可能原因：
1. 学习率太小 → 尝试 2e-4 或 5e-4
2. 数据格式有问题 → 打印 tokenize 后的数据确认格式正确
3. LoRA target_modules 没选对 → 确认加了 q_proj, v_proj
4. 数据量太少 → 至少需要几百条
```

### A.3 输出质量差

```
可能原因：
1. 训练数据质量不好 → 检查训练数据中的 output 是否正确、一致
2. 训练不充分 → 增加 epoch 或数据量
3. 过拟合 → 减少 epoch，加大 dropout
4. 推理参数不合适 → temperature 设低（0.1），do_sample=False
```

### A.4 bitsandbytes 安装问题（Windows）

```bash
# Windows 上 bitsandbytes 可能装不上，替代方案：
# 1. 用 Colab/AutoDL（Linux 环境）
# 2. 不用量化，直接 float16 加载（显存够的话）
# 3. 安装 Windows 兼容版本
pip install bitsandbytes-windows
```

---

## 附录 B：关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| `transformers` | HF 核心库，加载模型和 Tokenizer |
| `datasets` | 数据加载和处理库 |
| `peft` | 参数高效微调（LoRA/QLoRA） |
| `trl` | 提供 SFTTrainer 做指令微调 |
| `accelerate` | 多卡/混合精度训练加速 |
| `bitsandbytes` | 模型量化（4-bit/8-bit） |
| Pipeline | 一行代码推理 |
| AutoModelForCausalLM | 加载生成式语言模型 |
| SFTTrainer | 指令微调专用训练器 |
| LoRA | 低秩适配：冻结原模型，只训练小矩阵 |
| QLoRA | 量化 + LoRA：4-bit 加载 + LoRA 训练 |
| GGUF | llama.cpp/Ollama 使用的模型格式 |
| Chat Template | 对话格式化模板，每个模型不同 |
| merge_and_unload | 合并 LoRA 权重回基座模型 |

---

## 附录 C：学完后的能力自检

完成 7 天学习后，你应该能回答：

- [ ] `pipeline` 和手动加载模型的区别？什么时候用哪个？
- [ ] `AutoModelForCausalLM` vs `AutoModelForSequenceClassification` 的区别？
- [ ] `padding` 和 `attention_mask` 的作用？
- [ ] `Trainer` 和 `SFTTrainer` 各自适用什么场景？
- [ ] `TrainingArguments` / `SFTConfig` 中 `learning_rate`、`batch_size`、`epochs` 怎么调？
- [ ] LoRA 解决了什么问题？`r` 值越大越好吗？
- [ ] QLoRA 相比 LoRA 多了什么？什么时候需要用？
- [ ] 训练好的 LoRA 权重如何合并到基座模型？
- [ ] GGUF 是什么？Q4_K_M 和 F16 有什么区别？
- [ ] 从数据准备到 Ollama 部署的完整流程是什么？

全部能答上来 → 你已经具备独立完成语句提纯模型微调的能力，可以进入**第三阶段：数据构造与正式微调**。
