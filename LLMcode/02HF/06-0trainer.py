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