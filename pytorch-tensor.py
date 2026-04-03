"""
一，张量：多维数组
    PyTorch中的张量（Tensor）是一个多维数组，可以在GPU上进行高效的计算。它类似于NumPy的ndarray，但具有更强大的功能和更高的性能。
    PyTorch中的张量支持自动求导，可以方便地进行反向传播和优化。此外，PyTorch还提供了丰富的函数和工具来操作张量，包括数学运算、索引、切片、广播等。
    PyTorch中的张量可以在CPU和GPU之间无缝切换，使得在训练深度学习模型时能够充分利用硬件资源。PyTorch还提供了丰富的API来创建和操作张量，包括随机数生成、线性代数运算、统计函数等。
    总之，PyTorch中的张量是一个强大而灵活的数据结构，是构建和训练深度学习模型的基础。   
    1，一维数组：[1.0, 2.0, 3.0]
    2，二维数组：[[1.0, 2.0], [3.0, 4.0]]
    3，三维数组：[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    构成矩阵形式
二，关于张量的代码实现
    import torch

    # 创建一个一维张量
    tensor_1d = torch.tensor([1.0, 2.0, 3.0])
    print("一维张量：", tensor_1d)

    用字典来存储不同维度的张量
    tensors = {
        "1D": torch.tensor([1.0, 2.0, 3.0]),
        "2D": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
三，张量的属性
    1，shape：张量的形状，表示每个维度的大小
    2，dtype：张量的数据类型，如float32、int64等
    3，device：张量所在的设备，如CPU或GPU
    4，requires_grad：表示是否需要计算梯度，默认为False
    5，ndim：张量的维度数量
    6，size：张量的元素总数
四，张量的操作
    1，矩阵运算：乘法、加法、转置等
    2，索引和切片：访问张量的特定元素或子
"""
import torch
tensor =torch.tensor([[1,2],[3,4]])

#查看张量属性
print("张量的形状：", tensor.shape)
print("张量的数据类型：", tensor.dtype)

#基本算数运算
tensor_add = tensor + 2
print("张量加法：", tensor_add)

#矩阵乘法
tensor_mul = torch.matmul(tensor, tensor)
print("张量乘法：", tensor_mul)


"""
五，创建张量：empty()和rand()
    1，empty()：创建一个未初始化的张量，元素的值是随机的
    2，rand()：创建一个随机数张量，元素的值在0到1之间均匀分布(归一化思想，将大型数据集的数值缩放到0和1之间，便于模型训练和优化)
    3.zeros()：创建一个全零张量
    4.rand_like()：创建一个与给定张量形状相同的随机数张量
    5.torch.view()：改变张量的形状，但不改变其数据
"""
#创建一个未初始化的张量
empty_tensor = torch.empty(2, 3)
print("未初始化的张量：", empty_tensor)
#创建一个随机数张量
rand_tensor = torch.rand(2, 3)
print("随机数张量：", rand_tensor)
#创建一个全零张量
zeros_tensor = torch.zeros(2, 3)
print("全零张量：", zeros_tensor)
#创建一个与给定张量形状相同的随机数张量
rand_like_tensor = torch.rand_like(tensor, dtype=torch.float)
print("与给定张量形状相同的随机数张量：", rand_like_tensor)
#改变张量的形状，但不改变其数据
view_tensor = tensor.view(4)
print("改变形状的张量：", view_tensor)


"""
tensor张量与传统的numpy中的数组不同
六，将tensor转化个为numpy。array：
    tensor.numpy()：将PyTorch张量转换为NumPy数组
    torch.from_numpy()：将NumPy数组转换为PyTorch张量
七，CPU和GPU之间的张量转换
    1，to()：将张量移动到指定的设备上，如CPU或GPU
    2，cuda()：将张量移动到GPU上
"""
