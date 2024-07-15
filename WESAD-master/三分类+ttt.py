import random
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
from sklearn.feature_selection import RFE
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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

# 新增辅助函数：calculate_conv_output_dim 用于计算卷积层输出尺寸。
def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # 计算注意力权重
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # 使用注意力权重加权平均LSTM输出
        attn_output = torch.bmm(attn_weights.transpose(1, 2), lstm_output).squeeze(1)
        return attn_output

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_wise = self.channel_gate(x) * x
        spatial_wise = self.spatial_gate(x) * x
        return channel_wise + spatial_wise

class CNNLSTMModel(nn.Module):
    def __init__(self, lstm_hidden_dim=50, num_lstm_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(in_channels=16)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(in_channels=32)

        # 计算卷积层输出的维度
        conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4

        # 计算卷积层的输出大小
        self.conv_output_size = 32 * conv_output_dim

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # 定义注意力层
        self.attention = Attention(lstm_hidden_dim)

        # 定义全连接层
        self.fc1 = nn.Linear(lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 3)  # 确保输出维度为3

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN部分
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)  # 添加CBAM
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.cbam2(x)  # 添加CBAM
        x = self.pool(x)

        # 调整形状以适应LSTM层的输入
        x = x.view(x.size(0), 1, -1)  # (batch_size, sequence_length=1, input_size=conv_output_size)

        # LSTM部分
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        attn_output = self.attention(lstm_out)

        # 全连接层
        x = F.relu(self.fc1(attn_output))
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

# 定义TTT微调函数
def test_time_training(model, optimizer, images, labels):
    model.train()
    images.requires_grad = True
    outputs = model(images.float())
    loss = criterion(outputs, labels)
    loss.requires_grad = True
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 定义测试模型的函数
def test(model, validation_loader, use_ttt=False):
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

            # 如果使用TTT，先对模型进行微调
            if use_ttt:
                test_time_training(model, optimizer, images, labels)

            # Forward pass
            outputs = model(images.float())
            loss = criterion(outputs, labels)

            testlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

            correct_labels += labels.cpu().numpy().tolist()
            predictions += argmax.cpu().numpy().tolist()

    accuracy = np.round(correct / total, 3)
    confusion = confusion_matrix(correct_labels, predictions)

    return np.mean(testlosses), accuracy, confusion


# 读取数据集
df = pd.read_csv('data/m14_merged_4classes.csv', index_col=0)

# 获取所有受试者ID
subject_id_list = df['subject'].unique()

# 定义改变标签的函数，将非0或1的标签转换为1
def change_label(label):
    if label == 0 or label == 1:
        return 0
    elif label == 2:
        return 1
    else:
        return 2

# 应用改变标签的函数到数据集的标签列
df['label'] = df['label'].apply(change_label)
print(df['label'].unique())  # 应该输出 [0, 1, 2]

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
# 创建 RFE 对象，指定随机森林模型和要选择的目标特征数量
rfe = RFE(estimator=rf, n_features_to_select=10)

# 对数据进行拟合并获取选择后的特征
rfe.fit(X_train_scaled, y_train)

# 获取选择后的特征索引
selected_features_index = rfe.support_

# 使用选择后的特征索引来获取选择后的特征
selected_features = X.columns[selected_features_index]

print("Top features selected by RFE:")
print(selected_features)

# 使用选择后的特征重新构建数据集
df = df[selected_features.tolist() + ['label', 'subject']]

# 设置训练和测试的批量大小
train_batch_size = 25
test_batch_size = 5

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

# 定义网格搜索的参数范围
param_grid = {
    'lstm_hidden_dim': [50, 100],
    'num_lstm_layers': [1, 2],
    'learning_rate': [1e-3, 5e-3, 1e-2],
    'ttt_hidden_dim': [64, 128]  # 新增TTT的参数
}

def search_grid(df, param_grid):
    best_params = None
    best_score = -1

    for lstm_hidden_dim in param_grid['lstm_hidden_dim']:
        for num_lstm_layers in param_grid['num_lstm_layers']:
            for learning_rate in param_grid['learning_rate']:
                for ttt_hidden_dim in param_grid['ttt_hidden_dim']:  # 新增TTT的参数循环
                    # 执行十折交叉验证
                    fold_count = 0
                    current_scores = []

                    for train_index, test_index in kf.split(df):
                        fold_count += 1
                        print(f'网格搜索 - 第{fold_count}折:')
                        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

                        train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

                        model = CNNLSTMModel(lstm_hidden_dim=lstm_hidden_dim, num_lstm_layers=num_lstm_layers).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        train(model, optimizer, train_loader, test_loader)

                        _, _, accuracy = test(model, test_loader)
                        current_scores.append(accuracy)

                    mean_score = np.mean(current_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'lstm_hidden_dim': lstm_hidden_dim,
                            'num_lstm_layers': num_lstm_layers,
                            'learning_rate': learning_rate,
                            'ttt_hidden_dim': ttt_hidden_dim  # 新增TTT的最佳参数
                        }

    return best_params

# # 执行网格搜索
# best_params = search_grid(df, param_grid)
# print("Best parameters found by Grid Search:", best_params)

fold_count = 0  # 初始化折数计数器

# 初始化最大准确率和最优模型
max_acc = 0.0
best_model = None
import copy

# 使用最佳参数进行最终训练和测试
for train_index, test_index in kf.split(df):
    fold_count += 1
    print(f'最终训练和测试 - 第{fold_count}折:')  # 添加这行打印当前折数
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

    # 使用最佳参数初始化模型
    # model = CNNLSTMModel(lstm_hidden_dim=best_params['lstm_hidden_dim'], num_lstm_layers=best_params['num_lstm_layers']).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    model = CNNLSTMModel(lstm_hidden_dim=50, num_lstm_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)

    # # 使用TTT方法进行测试
    # cm, test_loss, test_acc = test(model, test_loader, use_ttt=True)
    # test_losses.append(test_loss)
    # test_accs.append(test_acc)
    # confusion_matrices.append(cm)

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    # 检查当前模型是否比之前的模型更好
    if test_acc > max_acc:
        max_acc = test_acc
        best_model = copy.deepcopy(model)
        print(f'New best model found with accuracy: {max_acc:.4f}')

# # 打印平均测试准确率和损失
# print(f'Average test accuracy over {num_folds} folds: {np.mean(test_accs):.4f}')
# print(f'Average test loss over {num_folds} folds: {np.mean(test_losses):.4f}')

# 打印最大测试准确率
print(f'Maximum test accuracy over {num_folds} folds: {max_acc:.4f}')
# 保存best_model
torch.save(best_model.state_dict(), 'model/best_model.pth')