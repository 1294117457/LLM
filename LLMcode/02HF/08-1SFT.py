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
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="cuda", # 修改这里，强行锁定第一张显卡
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
# 把 conversations 格式转成 text 字段
def format_to_text(example):
  example["text"] = tokenizer.apply_chat_template(
      example["conversations"], tokenize=False
  )
  return example
dataset = dataset.map(format_to_text)

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
    max_length=512,               # 最大序列长度
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