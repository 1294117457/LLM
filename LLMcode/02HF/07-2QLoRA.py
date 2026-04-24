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