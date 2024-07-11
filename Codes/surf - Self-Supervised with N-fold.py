import random
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('module://ipykernel.pylab.backend_inline')

# 自定义数据集类
class WESADDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop('subject', axis=1)
        self.labels = self.dataframe['label'].values
        self.dataframe.drop('label', axis=1, inplace=True)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].values
        x = x.reshape(1, -1)  # 调整输入形状为CNN
        y = self.labels[idx]
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.dataframe)

'''
# 获取数据加载器的函数
def get_data_loaders(df, train_subjects, test_subjects, train_batch_size=25, test_batch_size=5):
    # 根据随机选出来的人来分割训练集和测试集
    train_df = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)

    # 创建训练集和测试集的数据加载器
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=test_batch_size)

    return train_loader, test_loader
'''

# 新增辅助函数：calculate_conv_output_dim 用于计算卷积层输出尺寸
def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1

# 自监督学习模型
class SelfSupervisedCNNModel(nn.Module):
    def __init__(self):
        super(SelfSupervisedCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4
        self.fc1_input_dim = 32 * conv_output_dim
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 训练自编码器的函数
def train_autoencoder(model, data_loader, optimizer, criterion, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 训练模型的函数
def train(model, optimizer, train_loader, validation_loader, criterion, num_epochs=100):
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        valid_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.float())
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_acc = correct / total
        valid_loss = np.mean(valid_losses)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
    return history

# 测试模型的函数
def test(model, validation_loader, criterion):
    model.eval()
    test_losses = []
    correct = 0
    total = 0
    correct_labels = []
    predictions = []
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            correct_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    test_loss = np.mean(test_losses)
    accuracy = correct / total
    cm = confusion_matrix(correct_labels, predictions)
    return cm, test_loss, accuracy

# 读取数据集
df = pd.read_csv('C:\\Users\\asus\\Desktop\\m14_merged.csv', index_col=0)

# 获取所有受试者ID
subject_id_list = df['subject'].unique()

# 定义改变标签的函数
def change_label(label):
    if label == 0 or label == 1:
        return 0
    else:
        return 1

df['label'] = df['label'].apply(change_label)

# 特征选择
feats = ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max', 'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean', 'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean', 'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean', 'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height', 'weight', 'subject', 'label']

# 选择特征列
X = df[feats[:-2]]  # 去掉 'subject' 和 'label' 列
y = df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用训练集数据拟合scaler对象并进行转换
X_train_scaled = scaler.fit_transform(X_train)

# 使用同一个scaler对象转换测试集数据
X_test_scaled = scaler.transform(X_test)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 计算特征重要性
importances = rf.feature_importances_

# 将特征重要性转换为DataFrame
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# 按重要性排序
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 选择最重要的特征（例如前10个）
top_features = feature_importances['Feature'].iloc[:10].values

print("Top features selected by RandomForest:")
print(top_features)

# 使用最重要的特征重新构建数据集
df = df[top_features.tolist() + ['label', 'subject']]

# 设置训练和测试的批量大小
train_batch_size = 25
test_batch_size = 5

# 设置学习率
learning_rate = 5e-3

# 设置设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置训练的轮数
num_epochs = 100

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 初始化用于存储结果的列表
histories = []
confusion_matrices = []
test_losses = []
test_accs = []

# Sets the number of folds for cross validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_count = 0  # 初始化折数计数器

for train_index, test_index in kf.split(df):
    fold_count += 1  # 每一折开始时增加折数计数器
    print(f'第{fold_count}折:')  # 打印当前的折数
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=test_batch_size)

    # 训练自编码器
    input_dim = len(top_features)
    autoencoder = Autoencoder(input_dim).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    train_autoencoder(autoencoder, train_loader, ae_optimizer, nn.MSELoss(), num_epochs=50)

    # 每次迭代都重新初始化模型和优化器
    model = SelfSupervisedCNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    history = train(model, optimizer, train_loader, test_loader, criterion, num_epochs)
    histories.append(history)

    # 测试模型
    cm, test_loss, test_acc = test(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    confusion_matrices.append(cm)

# 打印平均测试准确率和损失
print(f'Average test accuracy over {num_folds} folds: {np.mean(test_accs):.4f}')
print(f'Average test loss over {num_folds} folds: {np.mean(test_losses):.4f}')
