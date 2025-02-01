import torch
from find_lr_dataset import data_loader, DvX_dataset_collate
from find_lr_loss_func import BCELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim
from torch_lr_finder import LRFinder  # 导入 torch-lr-finder

from model.model_ResNet_findLr import *
import matplotlib.pyplot as plt  # 确保正确导入 Matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
epochs = 72
input_shape = [224, 224]
learning_rate = 1e-7  # 学习率已设置为 5e-4
end_lr = 1e-5
batch_size = 32
num_iter = 800
train_annotation_path = './data/DvXray_train.txt'
val_annotation_path = './data/DvXray_test.txt'

checkpoint = None
from torch_lr_finder import TrainDataLoaderIter

def main():
    global checkpoint

    if checkpoint is None:
        model = AHCR(num_classes=15)
        parameters = model.parameters()
        optimizer = optim.AdamW(parameters, 
                            lr=learning_rate,  # 学习率
                            betas=(0.9, 0.999),  # 优化器 Betas
                            eps=1e-8,  # 数值稳定性参数
                            weight_decay=0.01)  # 权重衰减
    else:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
    model = model.to(device)
    criterion = BCELoss().to(device)

    # 训练集加载器
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    train_loader = DataLoader(data_loader(train_lines, input_shape), batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=DvX_dataset_collate, num_workers=batch_size, pin_memory=True)
    
    # 使用 torch-lr-finder 进行学习率范围搜索
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

    # 传入修改后的模型
    custom_train_loader = TrainDataLoaderIter(train_loader)
    lr_finder.range_test(custom_train_loader, end_lr=end_lr, num_iter=num_iter)
    lr_finder.plot()  # 生成学习率搜索结果图
    plt.savefig("lr_finder_plot.png")  # 保存为 PNG 文件

if __name__ == '__main__':
    main()
