import torch

a = torch.tensor([1.0,2.0,3.0])
print('a:',a)
print('a.shape:',a.shape)

b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print('b:',b)          # 3x2 的矩阵
print('b.shape:',b.shape)    # torch.Size([3, 2])

c = torch.randn(3, 4)   # 3行4列，值从标准正态分布采样
print('c:',c)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print('x + y:', x + y)       # 逐元素加法: tensor([5., 7., 9.])
print('x * y:', x * y)       # 逐元素乘法: tensor([4., 10., 18.])
print('x @ y:', x @ y)       # 点积(内积): tensor(32.)  → 1*4 + 2*5 + 3*6

# 矩阵乘法
A = torch.randn(3, 4)   # 3x4
B = torch.randn(4, 5)   # 4x5
C = A @ B                # 3x5  （矩阵乘法：前者列数 = 后者行数）
print('C.shape:', C.shape)           # torch.Size([3, 5])

print(torch.cuda.is_available())   # True 表示有 GPU

# 把 Tensor 放到 GPU 上
if torch.cuda.is_available():
    x_gpu = torch.randn(1000, 1000).cuda()
    y_gpu = torch.randn(1000, 1000).cuda()

    # 这个矩阵乘法在 GPU 上执行，比 CPU 快很多
    z_gpu = x_gpu @ y_gpu
    print('z_gpu.shape:', z_gpu.shape)   # torch.Size([1000, 1000])
    print('z_gpu.device:', z_gpu.device)  # cuda:0