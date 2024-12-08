import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import random
import numpy as np





# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载ResNet18并移除分类层
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
resnet18.eval()
resnet18.to(device)


# 路径设置
data_dir = "D:\\Mudestreda\\dataset_aug\\dataset_aug\\"
labels_path = "D:\\Mudestreda\\labels_aug\\labels_aug\\tool_distribution\\"

# 超参数
batch_size = 32


num_epochs = 5
num_classes = 3
patience = 10  # 早停的耐心次数，当验证集准确率连续5次不提升时停止训练


# 定义特征提取函数
def extract_features_from_image(image, model, device):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)  # 移动到设备
    with torch.no_grad():
        feature = model(image)
    return feature.view(-1)


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, device='cpu'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.device = device


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_ids = [row['id']]
        image_label = row['tool_label'] - 1  # 将标签转换为从0开始

        images = []
        for img_id in img_ids:
            img_paths = [os.path.join(self.img_dir, f"specX/{img_id}.png"),
                         os.path.join(self.img_dir, f"specY/{img_id}.png"),
                         os.path.join(self.img_dir, f"specZ/{img_id}.png"),
                         os.path.join(self.img_dir, f"tool/{img_id}.jpg")]
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]  # 转换为RGB
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            images.extend(imgs)

        image_features = []
        # 提取梅尔谱图的 ResNet18 特征
        for image in images:
            image_feature = extract_features_from_image(image, resnet18, self.device)
            image_features.append(image_feature)

        label = torch.tensor(image_label, dtype=torch.long, device=self.device)

        # 返回特征和分类标签
        return (
            torch.stack(image_features).to(self.device),
            label,
        )



# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])





# 基于 Transformer 的特征融合模型
class FeatureFusion(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_features=4, num_classes=3):
        super(FeatureFusion, self).__init__()

        # 共享的全连接层，用于降维
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)  # 增加 Batch Normalization

        # 分类层
        self.fc_concat = nn.Linear(reduced_dim * num_features, num_classes)  # 这里 num_features 是拼接的特征数

    def forward(self, features):
        # features_list 的形状是 (batch_size, num_features, feature_dim)
        # 对每个输入特征进行处理

        # 将每个特征通过共享的FC层降维
        features = self.shared_fc(features)  # 输出形状: (batch_size, reduced_dim)

        batch_size, seq_len, _ = features.shape
        features = features.view(-1, features.size(-1))  # 展平
        features = self.bn(features)  # BN
        features = features.view(batch_size, seq_len, -1)  # 恢复形状


        concatenated_features = features.view(batch_size, -1)
        # 将所有处理过的特征拼接起来，按特征维度拼接

        # 分类层
        output = self.fc_concat(concatenated_features)  # 输出形状: (batch_size, num_classes)

        return output.squeeze(-1)  # (batch_size)






# 在训练和验证时，将数据传输到正确的设备
test_dataset = CustomDataset(csv_file=os.path.join(labels_path, 'test.csv'), img_dir=data_dir, transform=transform, device=device)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = 512
reduced_dim = 256  # 降维到 128
model = FeatureFusion(input_dim=input_dim, reduced_dim=reduced_dim).to(device)


criterion = nn.CrossEntropyLoss()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


# 加载验证集最高准确率的模型
model.load_state_dict(torch.load("best_model4.pth"))

# 使用最佳模型进行测试集评估
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
