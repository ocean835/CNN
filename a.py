import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import psutil
import os
import torch.cuda as cuda
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取Cora数据集
file_path = 'D:/cora/cora.content'
data = pd.read_csv(file_path, sep='\t', header=None)
features = data.iloc[:, :-1].values #读取除了最后一列的所有列并转换为numpy数组
labels = data.iloc[:, -1].values #读取最后一列并转换为numpy数组

# 数据预处理
def preprocess_data(features, labels):
    max_values = np.max(features, axis=0) # 找到每一列的最大值
    max_values[max_values == 0] = 1
    features = features / max_values # 归一化处理
    features = torch.tensor(features, dtype=torch.float32) # 将特征转换为浮点型，并转换为PyTorch张量
    label_encoder = LabelEncoder() # 标签编码器，将字符串标签转换为数字标签
    labels = label_encoder.fit_transform(labels) # 将字符串标签转换为数字标签
    labels = torch.tensor(labels, dtype=torch.long) # 将标签转换为长整型，
    return features, labels

features, labels = preprocess_data(features, labels) # 预处理数据

# 划分训练集、验证集、测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42) # 0.1 是测试集的比例
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=42) # 1/9 是验证集的比例，训练集占 8/9，验证集占 1/9

# 创建 DataLoader
train_dataset = TensorDataset(X_train.clone().detach(), y_train.clone().detach()) # 使用 clone().detach() 防止梯度传播
val_dataset = TensorDataset(X_val.clone().detach(), y_val.clone().detach()) # 使用 clone().detach() 防止梯度传播
test_dataset = TensorDataset(X_test.clone().detach(), y_test.clone().detach())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 每次迭代打乱数据，增强训练效果，每次迭代使用的样本数量为32
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # 不打乱数据
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 不打乱数据

# 定义CNN模型
class SimpleCNNWithPooling(nn.Module):
    def __init__(self, input_features, num_classes): # 输入特征数和输出类别数
        super(SimpleCNNWithPooling, self).__init__() # 继承 nn.Module 类
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,3), stride=1, padding=(0,1)) # 输入通道数为 1，输出通道数为 32，卷积核大小为 3x3，步长为 1，填充为 1
        # padding=1 使得输出大小与输入大小相同
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))  # 最大池化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,3), stride=1, padding=(0,1)) # 输入通道数为 32，输出通道数为 64，卷积核大小为 3x3，步长为 1，填充为 1
        # padding=1 使得输出大小与输入大小相同
        self.pool2 = nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))  # 平均池化层
        self.fc1 = nn.Linear(64 * (input_features//4), 128) # 输入特征数为 64 * (input_features//4)，输出特征数为 128
        # 这里的 input_features//4 是因为经过两次池化后，特征图的大小减小了四分之一
        # 64 是第二个卷积层的输出通道数
        # 128 是全连接层的输出特征数
        self.fc2 = nn.Linear(128, num_classes) # 输入特征数为 128，输出特征数为 num_classes
        # num_classes 是分类的类别数
         
    def forward(self, x, show_pooling=False):
        x = x.unsqueeze(1).unsqueeze(2) # 添加通道维度和高度维度,以适应CNN的输入要求
        # x 的形状变为 (batch_size, 1, input_features, 1)
        # 这里的 1 是因为我们只处理一维数据，卷积操作需要四维输入，所以需要添加两个维度
      
        # 第一层卷积+池化
        x = torch.relu(self.conv1(x))
        if show_pooling:
            print("Before Pooling 1:", x.shape)
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.title("Before Pooling 1")
            plt.imshow(x[0,0].cpu().detach().numpy(), cmap='viridis')
        x = self.pool1(x)
        if show_pooling:
            print("After MaxPooling 1:", x.shape)
            plt.subplot(1,2,2)
            plt.title("After MaxPooling 1")
            plt.imshow(x[0,0].cpu().detach().numpy(), cmap='viridis')
            plt.show()
        
        # 第二层卷积+池化
        x = torch.relu(self.conv2(x))
        if show_pooling:
            print("Before Pooling 2:", x.shape)
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.title("Before Pooling 2")
            plt.imshow(x[0,0].cpu().detach().numpy(), cmap='viridis')
        x = self.pool2(x)
        if show_pooling:
            print("After AvgPooling 2:", x.shape)
            plt.subplot(1,2,2)
            plt.title("After AvgPooling 2")
            plt.imshow(x[0,0].cpu().detach().numpy(), cmap='viridis')
            plt.show()
        
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
# 模型实例化并移动到设备
input_features = features.shape[1] # 输入特征数
# 这里的 input_features 是特征的维度
model = SimpleCNNWithPooling(input_features=input_features, num_classes=7).to(device) #cuda有7个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8) # Adam 优化器

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()   # 记录训练开始时间
    process = psutil.Process(os.getpid()) # 获取当前进程信息

    for epoch in range(num_epochs): # 迭代训练
        model.train() # 设置模型为训练模式,启用 dropout 和 batch normalization
        running_loss = 0.0 # 初始化损失
        epoch_start_time = time.time() # 记录每个 epoch 的开始时间

        for inputs, labels in train_loader: # 遍历训练集
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad() # 将梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step()    # 更新参数
            running_loss += loss.item() # 累加损失

        epoch_end_time = time.time() # 记录每个 epoch 的结束时间
        epoch_time = epoch_end_time - epoch_start_time # 计算每个 epoch 的训练时间
        
        # 计算每个 epoch 的内存使用情况
        if cuda.is_available(): 
            memory_allocated = cuda.memory_allocated() / (1024 ** 2)  # 获取当前内存使用量
            max_memory_allocated = cuda.max_memory_allocated() / (1024 ** 2)  # 获取最大内存使用量
        else:
            memory_allocated = process.memory_info().rss / (1024 ** 2)  # 获取cpu内存使用量
            max_memory_allocated = memory_allocated

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, "  # 打印每个 epoch 的损失
              f"Time: {epoch_time:.2f}s, " # 打印每个 epoch 的训练时间
              f"Memory Allocated: {memory_allocated:.2f}MB, " # 打印当前内存使用量
              f"Max Memory Allocated: {max_memory_allocated:.2f}MB")  # 打印最大内存使用量
       
        # 每个 epoch 结束后在验证集上进行评估
        # 验证模型
        model.eval() # 设置模型为评估模式,禁用 dropout 和 batch normalization
        correct = 0 # 初始化正确预测的数量
        total = 0 # 初始化总样本数量
        # 在验证集上进行评估
        with torch.no_grad(): #不进行梯度计算
            for inputs, labels in val_loader: # 遍历验证集
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
                outputs = model(inputs) # 前向传播
                _, predicted = torch.max(outputs.data, 1) # 获取预测结果，返回最大值的索引
                total += labels.size(0) # 累加总样本数量
                # labels.size(0) 是每个 batch 的大小
                correct += (predicted == labels).sum().item() # 累加正确预测的数量
        # 计算验证集上的准确率

        print(f"Validation Accuracy: {100 * correct / total}%") # 打印验证集上的准确率

    end_time = time.time() # 记录训练结束时间
    total_time = end_time - start_time # 计算总训练时间
    print(f"Total Training Time: {total_time:.2f}s") # 打印总训练时间

train_model(model, train_loader, val_loader, criterion, optimizer) # 训练模型

# 测试模型
model.eval() # 设置模型为评估模式,禁用 dropout 和 batch normalization
correct = 0 # 初始化正确预测的数量
total = 0 # 初始化总样本数量
test_start_time = time.time() # 记录测试开始时间
process = psutil.Process(os.getpid()) # 获取当前进程信息

with torch.no_grad(): #不进行梯度计算
    for inputs, labels in test_loader: # 遍历测试集
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
        outputs = model(inputs) # 前向传播
        _, predicted = torch.max(outputs.data, 1)   # 获取预测结果，返回最大值的索引
        total += labels.size(0) # 累加总样本数量
        correct += (predicted == labels).sum().item() # 累加正确预测的数量

test_end_time = time.time() # 记录测试结束时间
test_time = test_end_time - test_start_time # 计算测试时间

if cuda.is_available(): 
    memory_allocated = cuda.memory_allocated() / (1024 ** 2) # 获取当前内存使用量
    max_memory_allocated = cuda.max_memory_allocated() / (1024 ** 2) # 获取最大内存使用量
else:
    memory_allocated = process.memory_info().rss / (1024 ** 2)
    max_memory_allocated = memory_allocated

print(f"Test Accuracy: {100 * correct / total}%") # 打印测试集上的准确率
print(f"Test Time: {test_time:.2f}s, " # 打印测试时间
      f"Memory Allocated: {memory_allocated:.2f}MB, " # 打印当前内存使用量
      f"Max Memory Allocated: {max_memory_allocated:.2f}MB") # 打印最大内存使用量