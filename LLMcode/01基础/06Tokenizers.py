from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

texts = [
    "保研",
    "清华大学计算机系的保研要求是什么",
    "我想了解清华大学计算机科学与技术系的推荐免试研究生申请条件和具体流程",
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
]

for text in texts:
    ids = tokenizer.encode(text)
    tokens = tokenizer.tokenize(text)
    print(f"[{len(ids):>3} tokens] {text}")
    print(f"           切分: {tokens}")
    print()