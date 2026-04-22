import torch
# 假设我们要学习一个简单的函数：y = w * x + b
# 其中 w 和 b 是要学习的参数

w = torch.tensor(2.0, requires_grad=True)   # requires_grad=True 告诉 PyTorch "追踪这个参数"
b = torch.tensor(1.0, requires_grad=True)

# 前向计算
x = torch.tensor(3.0)
y_pred = w * x + b      # 预测值 = 2*3+1 = 7
y_true = torch.tensor(10.0)  # 真实值是 10

# 计算误差（loss）
loss = (y_pred - y_true) ** 2   # 均方误差 = (7-10)² = 9
print(f"loss = {loss.item()}")  # 9.0

# 反向传播 — 自动算出 w 和 b 各自该怎么调
loss.backward()

print(f"w 的梯度 = {w.grad}")  # -18.0 → 说明 w 应该增大（梯度为负，往正方向调）
print(f"b 的梯度 = {b.grad}")  # -6.0  → 说明 b 也应该增大