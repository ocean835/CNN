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

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取Cora数据集
file_path = 'D:/cora/cora.content'
data = pd.read_csv(file_path, sep='\t', header=None)
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# 数据预处理
def preprocess_data(features, labels):
    max_values = np.max(features, axis=0) # 找到每一列的最大值
    max_values[max_values == 0] = 1
    features = features / max_values # 归一化处理
    features = torch.tensor(features, dtype=torch.float32) # 将特征转换为浮点型
    label_encoder = LabelEncoder() # 标签编码器
    labels = label_encoder.fit_transform(labels) # 将字符串标签转换为数字标签
    labels = torch.tensor(labels, dtype=torch.long) # 将标签转换为长整型
    return features, labels

features, labels = preprocess_data(features, labels)

# 划分训练集、验证集、测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42) # 0.1 是测试集的比例
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=42) # 1/9 是验证集的比例，训练集占 8/9，验证集占 1/9

# 创建 DataLoader
train_dataset = TensorDataset(X_train.clone().detach(), y_train.clone().detach()) # 使用 clone().detach() 防止梯度传播
val_dataset = TensorDataset(X_val.clone().detach(), y_val.clone().detach()) # 使用 clone().detach() 防止梯度传播
test_dataset = TensorDataset(X_test.clone().detach(), y_test.clone().detach())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 每次迭代打乱数据，增强训练效果
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # 不打乱数据
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 不打乱数据

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 输入通道数为 1，输出通道数为 32，卷积核大小为 3x3，步长为 1，填充为 1
        # padding=1 使得输出大小与输入大小相同
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 输入通道数为 32，输出通道数为 64，卷积核大小为 3x3，步长为 1，填充为 1
        # padding=1 使得输出大小与输入大小相同
        self.fc1 = nn.Linear(64 * input_features, 128) # 输入特征数为 64 * input_features，输出特征数为 128
        # 64 * input_features 是因为经过两次卷积后，特征图的大小不变
        self.fc2 = nn.Linear(128, num_classes) # 输入特征数为 128，输出特征数为 num_classes
        # num_classes 是分类的类别数

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(3) # 添加通道维度和高度维度
        # x 的形状变为 (batch_size, 1, input_features, 1)
        # 这里的 1 是因为我们只处理一维数据，卷积操作需要四维输入，所以需要添加两个维度
        x = torch.relu(self.conv1(x)) # 激活函数
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型实例化并移动到设备
input_features = features.shape[1]
model = SimpleCNN(input_features=input_features, num_classes=7).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8) # Adam 优化器

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if cuda.is_available():
            memory_allocated = cuda.memory_allocated() / (1024 ** 2)  # 转换为 MB
            max_memory_allocated = cuda.max_memory_allocated() / (1024 ** 2)  # 转换为 MB
        else:
            memory_allocated = process.memory_info().rss / (1024 ** 2)  # 转换为 MB
            max_memory_allocated = memory_allocated

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, "
              f"Time: {epoch_time:.2f}s, "
              f"Memory Allocated: {memory_allocated:.2f}MB, "
              f"Max Memory Allocated: {max_memory_allocated:.2f}MB")

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total}%")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f}s")

train_model(model, train_loader, val_loader, criterion, optimizer)

# 测试模型
model.eval()
correct = 0
total = 0
test_start_time = time.time()
process = psutil.Process(os.getpid())

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_end_time = time.time()
test_time = test_end_time - test_start_time

if cuda.is_available():
    memory_allocated = cuda.memory_allocated() / (1024 ** 2)
    max_memory_allocated = cuda.max_memory_allocated() / (1024 ** 2)
else:
    memory_allocated = process.memory_info().rss / (1024 ** 2)
    max_memory_allocated = memory_allocated

print(f"Test Accuracy: {100 * correct / total}%")
print(f"Test Time: {test_time:.2f}s, "
      f"Memory Allocated: {memory_allocated:.2f}MB, "
      f"Max Memory Allocated: {max_memory_allocated:.2f}MB")
