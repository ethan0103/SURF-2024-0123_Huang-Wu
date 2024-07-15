import random
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch_geometric
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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid

# from transformers import PretrainedConfig
# from transformers.activations import ACT2FN
# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
# )
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import ModelOutput, logging
# from transformers.utils.import_utils import is_causal_conv1d_available


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

# class TTT_layer(nn.Module):
#     def __init__(self):
#         self.task = Task()
#
#     def forward(self, in_seq):
#         state = Learner(self.task)
#         out_seq = []
#         for tok in in_seq:
#             state.train(tok)
#             out_seq.append(state.predict(tok))
#         return  out_seq
#
# class Task(nn.Module):
#     def __init__(self):
#         self.theta_K=nn.Param((d1, d2))
#         self.theta_V=nn.Param((d1, d2))
#         self.theta_Q=nn.Param((d1, d2))
#
#     def loss(self, f, x):
#         train_view = self.theta_K @ x
#         label_view = self.theta_V @ x
#         return MSE(f(train_view), label_view)
#
# class Learner():
#     def __init__(self,task):
#         self.task =task
#         # Linear here, but can be any modelself.model =Linear()# online GD here for simplicityself.optim=0GD()
#     def train(self, x):
#         # grad function wrt first arg
#         # of loss,which is self.model
#         grad_fn =grad(self.task.loss)
#         # calculate inner-loop grad
#         grad_in =grad_fn(self.model, x)
#
#         # starting from current params,
#         # step in direction of grad_in,
#         self.optim.step(self.model,grad_in)
# def predict(self, x):
#     test_view = self.task.theta_Q @ x
#     return self.model(test_view)


# class CNNLSTMTransformerModel(nn.Module):
#     def __init__(self, lstm_hidden_dim=50, num_lstm_layers=1, nhead=2, num_encoder_layers=2):
#         super(CNNLSTMTransformerModel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#
#         # 计算卷积层输出的维度
#         conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4
#
#         # 计算卷积层的输出大小
#         self.conv_output_size = 32 * conv_output_dim
#
#         # 定义LSTM层
#         self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
#
#         # 检查 lstm_hidden_dim 是否能被 nhead 整除
#         if lstm_hidden_dim % nhead != 0:
#             raise ValueError("lstm_hidden_dim must be divisible by nhead")
#
#         # 定义Transformer编码层
#         encoder_layers = TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=nhead)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
#
#         # 定义注意力层
#         self.attention = Attention(lstm_hidden_dim)
#
#         # 定义全连接层
#         self.fc1 = nn.Linear(lstm_hidden_dim, 128)
#         self.fc2 = nn.Linear(128, 3)  # 确保输出维度为3
#
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         # CNN部分
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#
#         # 调整形状以适应LSTM层的输入
#         x = x.view(x.size(0), 1, -1)  # (batch_size, sequence_length=1, input_size=conv_output_size)
#
#         # LSTM部分
#         lstm_out, _ = self.lstm(x)
#
#         # Transformer部分
#         transformer_out = self.transformer_encoder(lstm_out)
#
#         # 注意力机制
#         attn_output = self.attention(transformer_out)
#
#         # 全连接层
#         x = F.relu(self.fc1(attn_output))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return F.log_softmax(x, dim=1)
class CNNLSTMGNNModel(nn.Module):
    def __init__(self, input_size=100, lstm_hidden_dim=50, num_lstm_layers=1, gnn_hidden_dim=64, num_gnn_layers=2):
        super(CNNLSTMGNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 计算卷积层输出的维度
        conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4
        self.conv_output_size = 32 * conv_output_dim

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=lstm_hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True)

        # Define GNN layers
        self.gnn = GCNConv(lstm_hidden_dim, gnn_hidden_dim)
        self.num_gnn_layers = num_gnn_layers

        # Define attention layer
        self.attention = Attention(lstm_hidden_dim)

        # Define fully connected layers
        self.fc1 = nn.Linear(gnn_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN part
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Reshape to fit the LSTM layer input
        x = x.view(x.size(0), -1)  # (batch_size, conv_output_size)
        x = x.unsqueeze(1)  # (batch_size, sequence_length=1, input_size=conv_output_size)
        print(f"Shape after CNN and reshape: {x.shape}")

        # LSTM part
        lstm_out, _ = self.lstm(x)
        print(f"Shape after LSTM: {lstm_out.shape}")

        # Squeeze sequence dimension
        lstm_out = lstm_out.squeeze(1)  # (batch_size, lstm_hidden_dim)
        print(f"Shape after squeezing LSTM output: {lstm_out.shape}")

        # GNN part
        num_nodes = lstm_out.size(0)
        edge_index = grid(height=num_nodes, width=1, device=lstm_out.device).view(2, -1).long()
        print(f"Edge index shape: {edge_index.shape}")

        for _ in range(self.num_gnn_layers):
            lstm_out = F.relu(self.gnn(lstm_out, edge_index))
            print(f"Shape after GNN layer: {lstm_out.shape}")

        # Attention mechanism
        attn_output = self.attention(lstm_out.unsqueeze(1))
        print(f"Shape after attention: {attn_output.shape}")

        # Fully connected layers
        x = F.relu(self.fc1(attn_output))
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"Shape before output: {x.shape}")

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
    return cm, test_loss, accuracy

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
# param_grid = {
#     'lstm_hidden_dim': [50, 100],
#     'num_lstm_layers': [1, 2],
#     'learning_rate': [1e-3, 5e-3, 1e-2],
#     'nhead': [2, 4],
#     'num_encoder_layers': [1, 2]
# }
# 定义网格搜索的参数范围
param_grid = {
    'lstm_hidden_dim': [50, 100],
    'num_lstm_layers': [1, 2],
    'learning_rate': [1e-3, 5e-3, 1e-2],
    # 'nhead': [2, 4],
    # 'num_encoder_layers': [1, 2],
    'gnn_hidden_dim': [64, 128],  # Added GNN-related parameter
    'num_gnn_layers': [1, 2]  # Added GNN-related parameter
}

# 定义网格搜索的函数，不包括交叉验证的部分
def search_grid(df, param_grid):
    best_params = None
    best_score = -1

    for lstm_hidden_dim in param_grid['lstm_hidden_dim']:
        for num_lstm_layers in param_grid['num_lstm_layers']:
            for learning_rate in param_grid['learning_rate']:
                for gnn_hidden_dim in param_grid['gnn_hidden_dim']:
                    for num_gnn_layers in param_grid['num_gnn_layers']:
                            # Perform ten-fold cross-validation
                            fold_count = 0
                            current_scores = []

                            for train_index, test_index in kf.split(df):
                                fold_count += 1
                                print(f'网格搜索 - 第{fold_count}折:')
                                train_df, test_df = df.iloc[train_index], df.iloc[test_index]

                                train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

                                model = CNNLSTMGNNModel(lstm_hidden_dim=lstm_hidden_dim, num_lstm_layers=num_lstm_layers, gnn_hidden_dim=gnn_hidden_dim, num_gnn_layers=num_gnn_layers).to(device)
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
                                    'gnn_hidden_dim': gnn_hidden_dim,
                                    'num_gnn_layers': num_gnn_layers,
                                }

    return best_params

# 执行网格搜索
best_params = search_grid(df, param_grid)
print("Best parameters found by Grid Search:", best_params)

fold_count = 0  # 初始化折数计数器

# lstm_hidden_dim = 50
# num_lstm_layers = 1
# learning_rate = 0.001
# nhead = 2
# num_encoder_layers = 1


# 使用最佳参数进行最终训练和测试
for train_index, test_index in kf.split(df):
    fold_count += 1
    print(f'最终训练和测试 - 第{fold_count}折:')  # 添加这行打印当前折数
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

    model = CNNLSTMGNNModel(lstm_hidden_dim=best_params['lstm_hidden_dim'], num_lstm_layers=best_params['num_lstm_layers'], gnn_hidden_dim=best_params['gnn_hidden_dim'], num_gnn_layers=best_params['num_gnn_layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # model = CNNLSTMTransformerModel(lstm_hidden_dim=lstm_hidden_dim, num_lstm_layers=num_lstm_layers, nhead=nhead, num_encoder_layers=num_encoder_layers).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)

    cm, test_loss, test_acc = test(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    confusion_matrices.append(cm)

# 打印平均测试准确率和损失
print(f'Average test accuracy over {num_folds} folds: {np.mean(test_accs):.4f}')
print(f'Average test loss over {num_folds} folds: {np.mean(test_losses):.4f}')
