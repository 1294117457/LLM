import json
import time
from openai import OpenAI

# 1. API 配置
client = OpenAI(
    api_key="sk-7c7ee4a97ea048fe83eea9e301503cba", # 请确保这是你真实的、有额度的 Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2. 定义你想生成的领域和批次
DOMAINS = ["教育学习","计算机专业考研", "金融专业保研", "物理专业出国", "医学专业考研","求职招聘","法律咨询", "美食烹饪","数码科技"]
BATCHES_PER_DOMAIN = 2 # 测试时先设小点，比如每个领域只请求 2 次
BATCH_SIZE = 5         # 每次请求让大模型生成 5 条

# 核心提示词：教大模型“如何为你生成训练数据”
SYSTEM_PROMPT = """你是一个训练数据生成器。
任务：按用户指定的领域，生成大学升学相关的用户问句及其意图解析。
要求：严格返回包含 {BATCH_SIZE} 个对象的 JSON 数组。不要返回任何 Markdown 标记或多余的文字。
JSON 格式需严格如下（务必包含 "conversations" 字段以符合 SFT 训练标准）：
[
  {{
    "conversations": [
      {{"role": "system", "content": "请将以下用户问句提取为结构化意图JSON"}},
      {{"role": "user", "content": "你想出的一条逼真的人类提问"}},
      {{"role": "assistant", "content": "{{\"intent\":\"提取的意图\", \"entities\":{{\"school\":\"学校名\", \"major\":\"专业名\", \"type\":\"升学方式\"}}}}"}}
    ]
  }}
]"""

# 3. 循环调用生成
all_data = []
MAX_RETRIES = 3

print("开始生成数据...")
for domain in DOMAINS:
    print(f"\n--- 正在生成领域：{domain} ---")
    for batch_idx in range(BATCHES_PER_DOMAIN):
        print(f"  请求批次 {batch_idx + 1}/{BATCHES_PER_DOMAIN}...")
        
        user_prompt = f"请生成 {BATCH_SIZE} 条关于【{domain}】领域的数据，提问口语化，意图分类精准。"

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.format(BATCH_SIZE=BATCH_SIZE)},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.8, # 让生成的提问更有随机性和多样性
                )
                
                # 尝试解析返回的 JSON 字符串
                content = response.choices[0].message.content.strip()
                # 剔除可能带有的 markdown code block 外衣
                if content.startswith("```json"): content = content[7:]
                if content.startswith("```"): content = content[3:]
                if content.endswith("```"): content = content[:-3]
                
                batch_data = json.loads(content.strip())
                all_data.extend(batch_data)
                
                print(f"    成功获取 {len(batch_data)} 条。")
                break # 成功则跳出重试循环

            except Exception as e:
                print(f"    重试 {attempt + 1}/{MAX_RETRIES} 失败: {e}")
                time.sleep(3)

output_file = "sft_train_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print(f"\n全部生成完毕！共 {len(all_data)} 条数据，已保存至 {output_file}")

# ========================
# 5. 随机抽样验证数据质量
# ========================
import random

print("\n==============================")
print("开始进行人工抽样验证...")
print("==============================\n")

# 加载刚生成的数据进行抽样（这里抽样 5 条，如果你的数据量够大可以设为 50）
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

sample_size = min(5, len(data)) 
samples = random.sample(data, sample_size)

for i, s in enumerate(samples):
    conversations = s.get("conversations", [])
    
    user_input = ""
    assistant_output = ""
    
    # 从 conversations 数组中提取 user 和 assistant 的对话
    for msg in conversations:
        if msg.get("role") == "user":
            user_input = msg.get("content")
        elif msg.get("role") == "assistant":
            assistant_output = msg.get("content")
            
    print(f"--- 样本 {i+1}/{sample_size} ---")
    print(f"输入 (User): {user_input}")
    print(f"输出 (Assistant): {assistant_output}")
    print("-" * 30 + "\n")
    # 你可以通过终端输出人工判断：大模型自动生成的问答对不对？需不需要修改 System Prompt？