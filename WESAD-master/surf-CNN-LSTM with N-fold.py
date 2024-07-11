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
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 定义一个自定义的数据集类，用于封装数据集
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


# 定义特征列表
feats = ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
         'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
         'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
         'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean',
         'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
         'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height',
         'weight', 'subject', 'label']

# 定义获取数据加载器的函数
def get_data_loaders(df, train_subjects, test_subjects, train_batch_size=25, test_batch_size=5):

    # 根据随机选出来的人来分割训练集和测试集
    train_df = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)

    # 创建训练集和测试集的数据加载器
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size)

    return train_loader, test_loader

#新增辅助函数：calculate_conv_output_dim 用于计算卷积层输出尺寸。
def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1

# CNN 模型，CNN可以共享参数
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNLSTMModel, self).__init__()

        # 定义CNN层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 计算卷积层输出的维度
        conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4

        # LSTM层
        self.lstm = nn.LSTM(input_size=32 * conv_output_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 执行CNN层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 调整形状以适应LSTM输入
        x = x.permute(0, 2, 1)

        # 执行LSTM层
        lstm_out, _ = self.lstm(x)

        # 取LSTM输出的最后一个时间步
        x = lstm_out[:, -1, :]

        # 执行全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# 定义训练模型的函数
def train(model, optimizer, train_loader, validation_loader):
    # 初始化记录训练和验证过程中的损失和准确率的字典
    history = {'train_loss': {}, 'train_acc': {}, 'valid_loss': {}, 'valid_acc': {}}
    # 训练模型
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        trainlosses = []

        for batch_index, (images, labels) in enumerate(train_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

        history['train_loss'][epoch] = np.mean(trainlosses)
        history['train_acc'][epoch] = correct / total

        if epoch % 10 == 0:
            with torch.no_grad():

                losses = []
                total = 0
                correct = 0

                for images, labels in validation_loader:
                    #
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images.float())
                    loss = criterion(outputs, labels)

                    # Compute accuracy
                    _, argmax = torch.max(outputs, 1)
                    correct += (labels == argmax).sum().item()  # .mean()
                    total += len(labels)

                    losses.append(loss.item())

                history['valid_acc'][epoch] = np.round(correct / total, 3)
                history['valid_loss'][epoch] = np.mean(losses)

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(losses):.4}, Acc: {correct / total:.2}')

    return history

# 定义测试模型的函数
def test(model, validation_loader):
    print('Evaluating model...')
    # Test
    model.eval()

    total = 0
    correct = 0
    testlosses = []
    correct_labels = []
    predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validation_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # 计算实际概率
            probabilities = torch.exp(outputs)

            '''
            # 打印概率
            for i, prob in enumerate(probabilities):
                print(f"Sample {i + 1}: Class 0 Probability = {prob[0]:.4f}, Class 1 Probability = {prob[1]:.4f}")
            '''

            # Loss
            loss = criterion(outputs, labels)

            testlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

            correct_labels.extend(labels)
            predictions.extend(argmax)

    test_loss = np.mean(testlosses)
    accuracy = np.round(correct / total, 2)
    print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')

    y_true = [label.item() for label in correct_labels]
    y_pred = [label.item() for label in predictions]

    cm = confusion_matrix(y_true, y_pred)
    # TODO: return y true and y pred, make cm after ( use ytrue/ypred for classification report)
    # return [y_true, y_pred, test_loss, accuracy]
    return cm, test_loss, accuracy

# 读取数据集
df = pd.read_csv('./data/m14_merged.csv', index_col=0)
# 获取所有受试者ID
subject_id_list = df['subject'].unique()


# 定义改变标签的函数，将非0或1的标签转换为1
def change_label(label):
    if label == 0 or label == 1:
        return 0
    else:
        return 1


# 应用改变标签的函数到数据集的标签列
df['label'] = df['label'].apply(change_label)


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

# 按照重要性排序
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

# 设置交叉验证的折数
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_count = 0  # 初始化折数计数器

for train_index, test_index in kf.split(df):
    fold_count += 1  # 每一折开始时增加折数计数器
    print(f'第{fold_count}折:')  # 打印当前的折数
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

    # 每次迭代都重新初始化模型和优化器
    model = CNNLSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)

    cm, test_loss, test_acc = test(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    confusion_matrices.append(cm)

# 打印平均测试准确率和损失
print(f'Average test accuracy over {num_folds} folds: {np.mean(test_accs):.4f}')
print(f'Average test loss over {num_folds} folds: {np.mean(test_losses):.4f}')
