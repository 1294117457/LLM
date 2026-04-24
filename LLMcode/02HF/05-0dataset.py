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