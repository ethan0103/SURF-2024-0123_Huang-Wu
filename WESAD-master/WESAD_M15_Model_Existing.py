# 导入操作系统库，用于文件路径操作等
import os
# 导入pickle模块，用于序列化和反序列化Python对象结构
import pickle
# 导入numpy库，用于科学计算
import numpy as np
# 导入pandas库，用于数据分析
import pandas as pd
# 导入matplotlib.pyplot，用于绘图
import matplotlib.pyplot as plt
# 导入seaborn库，用于制作更高级的统计图表
import seaborn as sns
# 设置matplotlib的后端，以便在Jupyter Notebook中显示图表
# %matplotlib inline
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')
# 再次导入matplotlib.pyplot，以确保可以绘图
import matplotlib.pyplot as plt

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入警告处理模块，设置警告级别为忽略
import warnings
warnings.filterwarnings('ignore')
# 从PyTorch中导入数据集和数据加载器工具
from torch.utils.data import Dataset
# 从sklearn.metrics中导入混淆矩阵函数
from sklearn.metrics import confusion_matrix

# 定义一个自定义的数据集类，用于封装数据集
class WESADDataset(Dataset):
    def __init__(self, dataframe):
        # 将数据集中的特征数据存储为一个属性
        self.dataframe = dataframe.drop('subject', axis=1)
        # 将标签数据存储为一个属性
        self.labels = self.dataframe['label'].values
        # 从数据集中删除标签列
        self.dataframe.drop('label', axis=1, inplace=True)

    def __getitem__(self, idx):
        # 根据索引获取特征数据和标签，并返回它们作为张量
        x = self.dataframe.iloc[idx].values
        y = self.labels[idx]
        return torch.Tensor(x), y

    def __len__(self):
        # 返回数据集的长度
        return len(self.dataframe)

# 定义特征列表
feats =   ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
           'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
           'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
           'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean',
           'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
           'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height',
           'weight','subject', 'label']
# 计算第一层网络的维度
layer_1_dim = len(feats) -2
print(layer_1_dim)

# 定义获取数据加载器的函数
def get_data_loaders(df, subject_id, train_batch_size=25, test_batch_size=5):
    # 读取数据集
    # df = pd.read_csv('data/m14_merged.csv', index_col=0)[feats]

    # 根据subject_id分割训练集和测试集
    train_df = df[df['subject'] != subject_id].reset_index(drop=True)
    test_df = df[df['subject'] == subject_id].reset_index(drop=True)

    # 创建训练集和测试集的数据加载器
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size)

    return train_loader, test_loader

# 定义神经网络模型
class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()
        # 定义模型的前馈网络结构
        self.fc = nn.Sequential(
            nn.Linear(29, 128),  # 第一层全连接层，输入29个特征，输出128个特征
            nn.ReLU(),           # ReLU激活函数
            nn.Linear(128, 256), # 第二层全连接层，输入128个特征，输出256个特征
            nn.ReLU(),           # ReLU激活函数
            nn.Linear(256, 2),   # 第三层全连接层，输出2个特征，对应两个类别
            nn.LogSoftmax(dim=1) # 对输出使用对数Softmax函数，用于多分类问题
        )

    def forward(self, x):
        # 定义模型的前向传播过程
        return self.fc(x)

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
df = pd.read_csv('data/m14_merged.csv', index_col=0)
# 获取所有受试者ID
subject_id_list = df['subject'].unique()
# 查看数据集的前几行
df.head()

# 定义改变标签的函数，将非0或1的标签转换为1
def change_label(label):
    if label == 0 or label == 1:
        return 0
    else:
        return 1

# 应用改变标签的函数到数据集的标签列
df['label'] = df['label'].apply(change_label)

# 选择特征列
df = df[feats]

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
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 初始化用于存储结果的列表
histories = []
confusion_matrices = []
test_losses = []
test_accs = []

# 对每个受试者执行留一法交叉验证
for _ in subject_id_list:
    print('\nSubject: ', _)
    model = StressNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_data_loaders(df, _)

    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)

    cm, test_loss, test_acc = test(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    confusion_matrices.append(cm)

# 打印平均测试准确率和损失
print(np.mean(test_accs))
print(np.mean(test_losses))

# 打印标签的分布情况
print(df['label'].value_counts())

# 绘制测试准确率的条形图
plt.figure(figsize=(14, 6))
plt.title('Testing Accuracies in Leave One Out Cross Validation by Subject Left Out as Testing Data')
sns.barplot(x=subject_id_list, y=test_accs)

# 绘制测试损失的条形图
plt.figure(figsize=(14, 3))
plt.title('Testing Losses in Leave One Out Cross Validation by Subject Left Out as Testing Data')
sns.barplot(x=subject_id_list, y=test_losses)

# 绘制混淆矩阵的热图
plt.figure(figsize=(15, 10))
for i in range(15):
    plt.subplot(4, 5, i + 1)
    cm = confusion_matrices[i]
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title(f'S{subject_id_list[i]}')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
plt.tight_layout()

# 注释掉的代码用于生成分类报告，但这部分代码未被执行
#from sklearn.metrics import classification_report

#target_names = ['Amusement', 'Baseline', 'Stress']
#print(classification_report(y_true, y_pred, target_names=target_names))
#%%
#torch.save(model.state_dict(), 'm13_model.pt')