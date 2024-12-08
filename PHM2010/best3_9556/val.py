import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


# 定义特征提取函数
def extract_features_from_image(image, model):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = model(image)
    return feature.view(-1)


# 数据集类定义
class SensorDataset(Dataset):
    def __init__(self, path, set_ids):
        """
        初始化数据集
        :param path: PHM2010根路径
        :param set_ids: 数据集的分组编号（列表形式，例如：[1, 4]）
        """
        self.path = path
        self.set_ids = set_ids
        self.label_files = {}

        # 预加载所有标签文件并存储到字典
        for set_id in set_ids:
            label_file = os.path.join(path, f'c{set_id}', f'c{set_id}_wear.csv')
            if os.path.exists(label_file):
                self.label_files[set_id] = pd.read_csv(label_file, header=0)
            else:
                raise FileNotFoundError(f"标签文件未找到: {label_file}")

        # 记录所有样本的总数量
        self.total_samples = sum(len(df) for df in self.label_files.values())

    def __len__(self):
        """
        返回数据集样本数量
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        获取指定索引的样本数据
        :param idx: 索引值
        :return: 传感器数据和对应分类标签
        """
        # 确定样本属于哪个 set_id
        cumulative_idx = 0
        for set_id, label_df in self.label_files.items():
            if idx < cumulative_idx + len(label_df):
                row = label_df.iloc[idx - cumulative_idx]  # 获取行数据
                break
            cumulative_idx += len(label_df)

        # 自动生成传感器文件名
        sample_id = int(row['cut'])  # 将 `cut` 字段的值转为整数

        file_name = f"c{set_id}\\c{set_id}\\c_{set_id}_{sample_id:03d}.csv"  # 补足三位
        sensor_file = os.path.join(self.path, file_name)

        # 加载传感器数据
        if os.path.exists(sensor_file):
            sensor_data_df = pd.read_csv(sensor_file, header=None)
            sensor_data = sensor_data_df.iloc[:, :7].values  # 假设取前两列为传感器数据
        else:
            raise FileNotFoundError(f"文件未找到: {sensor_file}")

        # 根据 sample_id 定义分类标签
        if 0 <= sample_id <= 26:
            label = 0
        elif 27 <= sample_id <= 208:
            label = 1
        else:
            label = 2

        # 生成梅尔谱图
        mel_images = []
        for col in sensor_data.T:
            mel_image = self.generate_mel_image(col)
            mel_images.append(mel_image)

        # 提取梅尔谱图的 ResNet18 特征
        mel_features = []
        for mel_image in mel_images:
            mel_feature = extract_features_from_image(mel_image, resnet18)
            mel_features.append(mel_feature)

        # 返回特征和分类标签
        return (
            torch.stack(mel_features),
            torch.tensor(label, dtype=torch.long),  # 标签作为整数分类
        )

    def generate_mel_image(self, data, sr=50000):
        """
        根据传感器数据生成梅尔谱图
        :param data: 传感器数据
        :param sr: 采样率 (50,000 Hz)
        :return: 梅尔谱图图像
        """
        # 参数调整
        n_fft = 4000  # 20 毫秒窗口长度
        hop_length = 2000
        n_mels = 512  # 梅尔滤波器组数量

        # 生成梅尔谱图
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 绘制梅尔谱图到内存
        fig, ax = plt.subplots(figsize=(5, 5))
        librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
        plt.axis('off')

        # 保存图像到内存中的字节流
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        # 将字节流转换为 PIL 图像
        mel_image = Image.open(buf).convert('RGB')
        return mel_image


# 定义Transformer模型，增加共享的FC层
class FeatureFusionTransformer(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_layers=2, nhead=4):
        super(FeatureFusionTransformer, self).__init__()

        # 共享的全连接层，用于降维
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)  # 增加 Batch Normalization

        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 回归层
        self.regressor = nn.Linear(reduced_dim, 3)

    def forward(self, features):
        # 将每个 Mel 谱图特征和工具特征传入共享的FC层
        features = self.shared_fc(features)

        # 在批次维度上应用 Batch Normalization
        # 需要先调整形状 (batch_size, seq_len, reduced_dim) -> (batch_size * seq_len, reduced_dim) 才能应用 BN
        batch_size, seq_len, _ = features.shape
        features = features.view(-1, features.size(-1))  # 展平
        features = self.bn(features)  # BN
        features = features.view(batch_size, seq_len, -1)  # 恢复形状

        # 通过 Transformer 编码器
        transformer_output = self.transformer_encoder(features)

        # 使用 Transformer 输出的平均值进行回归
        output = transformer_output[:, 2, :]  # 改为 dim=1 来保持 batch 维度

        # 预测磨损值
        wear_prediction = self.regressor(output)

        # 去除多余的维度，将形状变为 (batch_size)
        wear_prediction = wear_prediction.squeeze(-1)  # (batch_size)

        return wear_prediction




# 加载ResNet18并移除分类层
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
resnet18.eval()

# 配置路径
path = "D:\\PHM2010\\"  # 替换为实际路径

val_set_ids = [6]

# 实例化数据集
val_dataset = SensorDataset(path, val_set_ids)


# 查看数据集长度
# print(f"数据集包含样本数: {len(dataset)}")

# 创建 DataLoader
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)



# 初始化模型、损失函数和优化器
input_dim = 512
reduced_dim = 128  # 降维到 128
model = FeatureFusionTransformer(input_dim=input_dim, reduced_dim=reduced_dim)
model.load_state_dict(torch.load("best_model11.pth"))
model.eval()

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()


def calculate_accuracy(predictions, targets):
    """
    计算准确率
    :param predictions: 模型的输出，形状为 (batch_size, num_classes)
    :param targets: 真实标签，形状为 (batch_size)
    :return: 准确率
    """
    _, predicted_classes = torch.max(predictions, dim=1)  # 获取每个样本预测的类别
    correct = (predicted_classes == targets).sum().item()  # 计算正确预测的数量p
    total = targets.size(0)  # 总样本数量
    return correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_val = 0  # 正确预测数量
    total_val = 0    # 总样本数量

    with torch.no_grad():
        for features, wear_value in dataloader:
            features, wear_value = features.to(device), wear_value.to(device)

            # 前向传播
            wear_pred = model(features)

            # 计算损失
            loss = criterion(wear_pred, wear_value)
            running_loss += loss.item() * wear_value.size(0)

            # 计算验证准确率
            _, predicted_classes = torch.max(wear_pred, dim=1)
            correct_val += (predicted_classes == wear_value).sum().item()
            total_val += wear_value.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy = correct_val / total_val
    return val_loss, val_accuracy

# 验证集评估
val_loss, val_accuracy = validate(model, val_loader, criterion, device)
print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")



