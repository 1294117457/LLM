import torch

# 目标：让模型学会 y = 3x + 2 这个函数

# 训练数据
X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
Y = torch.tensor([5.0, 8.0, 11.0, 14.0, 17.0])   # y = 3x + 2

# 随机初始化参数（模型一开始不知道答案）
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01  # 学习率：每次调整的步幅

print("训练前: w={:.2f}, b={:.2f}".format(w.item(), b.item()))

for epoch in range(100):
    # 前向传播Y_pred = model(X)
    
    Y_pred = w * X + b

    # 计算误差loss = loss_fn(Y_pred, Y)
    loss = ((Y_pred - Y) ** 2).mean()

    # 反向传播loss.backward()
    loss.backward()

    # 更新参数（手动梯度下降）optimizer.step() 
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度（PyTorch 会累加梯度，所以每轮要清零）optimizer.zero_grad()   
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print("\n训练后: w={:.2f}, b={:.2f}".format(w.item(), b.item()))