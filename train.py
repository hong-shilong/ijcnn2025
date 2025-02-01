import argparse
import time  # 导入 time 模块

import torch
import torch.nn as nn
from thop import clever_format
from thop import profile
from timm.data import Mixup
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm

from dataset import data_loader, DvX_dataset_collate
from get_ap import AveragePrecisionMeter, AveragePrecisionMeter2
from loss_func import BCELoss
from model.model_v2 import *
from opt.ema import ModelEMA
from opt.warmup import LinearWarmup
from tools.log_to_tensorboard import *
from tools.result_to_csv import *

model_name = f'model_v2'
start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
log_head = f"Run time in {start_time}, Model:{model_name}"
update_content = log_head + f"更新内容：实验-model_ResNext_dual\n"
base_dir = f'./log/{model_name}_{start_time}/'
print(start_time)
depth_mult = 3
expansion = 1
# if 'Res' in model_name:
#     expansion = 1/4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_amp = True
use_ema = True
mixup = False

# Training parameters
start_epoch = 0
epochs = 64
input_shape = [256, 256]
batch_size = 64
num_workers = 32
warmup_duration = epochs // 10 * 175
# warmup_duration = 0
# learning_rate = 5e-5 * batch_size / 128
learning_rate = 5e-5

weight_decay = 1e-3

grad_clip = 5
max_norm = grad_clip
train_annotation_path = './data/DvXray_train.txt'
val_annotation_path = './data/DvXray_val.txt'
test_annotation_path = './data/DvXray_test.txt'

best_acc = 0
best_epoch = 0

# loss_Set
alpha = 0.7
threshold = 0.5


def save_checkpoint(epoch, model, optimizer, ap, warmup_scheduler, scheduler):
    global best_acc, start_time, best_epoch
    if ap > best_acc:
        best_acc = ap
        best_epoch = epoch
        state = {'epoch': epoch,
                 'model': model,
                 'optimizer': optimizer,
                 'start_time': start_time,
                 'best_acc': best_acc,
                 'best_epoch': best_epoch,
                 'warmup_scheduler': warmup_scheduler,
                 'scheduler': scheduler
                 }
        torch.save(state, base_dir + f'{model_name}_last.pth')
        torch.save(state, base_dir + f'{model_name}_best.pth')
    else:
        state = {'epoch': epoch,
                 'model': model,
                 'optimizer': optimizer,
                 'start_time': start_time,
                 'best_acc': best_acc,
                 'best_epoch': best_epoch,
                 'warmup_scheduler': warmup_scheduler,
                 'scheduler': scheduler
                 }
        torch.save(state, base_dir + f'{model_name}_last.pth')

    print(f'best in epoch {best_epoch + 1}, best_acc {best_acc}')

    # if epoch < 20 or epoch % 2 == 0:
    #     return
    # state = {'epoch': epoch,
    #          'model': model,
    #          'optimizer': optimizer,
    #          'start_time': start_time,
    #           'best_acc': best_acc,
    #          }
    # filename = 'ep%03d_ResNet_checkpoint.pth.tar' % (epoch + 1)
    # torch.save(state, './checkpoint/' + filename)


def main():
    args = parse_args()  # 获取命令行参数
    global start_epoch, start_time, num_workers, new_lr, only_eval, depth_mult, expansion, base_dir
    checkpoint = args.checkpoint  # 从命令行获取 checkpoint
    only_eval = args.eval  # 从命令行获取 only_eval

    if checkpoint is None:
        model = MyModel(num_classes=15, depth_mult=depth_mult, expansion=expansion)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)  # 使用 DataParallel
        model = model.to(device)
        flops, params = profile(model, (torch.randn(1, 3, 256, 256).to(device), torch.randn(1, 3, 256, 256).to(device)))
        flops, params = clever_format([flops, params], "%.3f")
        print(flops)
        print(params)
        parameters = model.parameters()
        optimizer = optim.AdamW(parameters,
                                lr=learning_rate,  # 学习率
                                betas=(0.9, 0.999),  # 优化器 Betas
                                eps=1e-8,  # 数值稳定性参数
                                weight_decay=weight_decay)  # 权重衰减
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
        # scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        warmup_scheduler = LinearWarmup(scheduler, warmup_duration)
    else:
        print(f"load checkpoints {checkpoint}")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        start_time = checkpoint['start_time']
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['best_epoch']
        warmup_scheduler = checkpoint['warmup_scheduler']
        scheduler = checkpoint['scheduler']
        base_dir = f'./log/{model_name}_{start_time}/'
    model = model.to(device)
    criterion = BCELoss().to(device)
    scaler = None
    ema = None

    # 训练集和验证集加载器
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    with open(test_annotation_path) as f:
        test_lines = f.readlines()
    test_loader = DataLoader(data_loader(test_lines, input_shape, is_train=False),
                             batch_size=batch_size // 4,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=DvX_dataset_collate,
                             num_workers=num_workers,
                             pin_memory=True)
    if only_eval:
        evaluate(test_loader, model, criterion, is_test=True)
        return 0
    train_loader = DataLoader(data_loader(train_lines, input_shape),
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              collate_fn=DvX_dataset_collate,
                              num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(data_loader(val_lines, input_shape, is_train=False),
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=DvX_dataset_collate,
                            num_workers=num_workers,
                            pin_memory=True)

    if use_ema:
        ema = ModelEMA(model)
    if use_amp:
        # 初始化 GradScaler
        scaler = torch.amp.GradScaler('cuda')
    mixup_fn = None
    if mixup:
        # Mixup 和 Cutmix 配置
        mixup_alpha = 0.8  # 设置 mixup 的 alpha 参数
        cutmix_alpha = 1.0  # 设置 cutmix 的 alpha 参数
        cutmix_minmax = None  # 设置 cutmix 的 minmax 参数
        mixup_prob = 1.0  # 设置 mixup 的执行概率
        mixup_switch_prob = 0.5  # 设置 mixup 和 cutmix 之间的切换概率
        mixup_mode = 'batch'  # 设置 mixup 的应用模式

        # 创建 Mixup 函数对象
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=0.1,  # 如果使用标签平滑（根据需要调整）
            num_classes=15  # 类别数，调整为你的实际类别数
        )

    writer = init_tensorboard(log_dir=base_dir)
    csv_filename = init_csv(base_dir + f'results.csv')

    # 打开日志文件，以追加模式写入
    log_file = open(base_dir + f'log.txt', 'a')
    if checkpoint is None:
        log_file.write(update_content)
        # 写入配置信息
        config_info = f"""Current Configuration:
        Model Name: {model_name}
        params: {params}
        flops: {flops}
        depth_mult:{depth_mult}
        expansion:{expansion}
        Start Epoch: {start_epoch}
        Total Epochs: {epochs}
        Input Shape: {input_shape}
        Batch Size: {batch_size}
        Num Workers: {num_workers}
        Learning Rate: {learning_rate}
        weight_decay: {weight_decay}
        Grad Clip: {grad_clip}
        Training Annotation Path: {train_annotation_path}
        Validation Annotation Path: {val_annotation_path}
        Only Eval Mode: {only_eval}
        Checkpoint: {checkpoint}
        use_ema: {use_ema}
        use_amp: {use_amp}
        alpha: {alpha}\n"""

        log_file.write(config_info)
        log_file.flush()  # 强制刷新缓存
    # warmup_scheduler.step()  # 更新 warmup
    # scheduler.step()
    print("training start")
    for epoch in range(start_epoch, epochs):
        # if epoch != 0 and epoch % 10 == 0:
        #     adjust_learning_rate(optimizer, 0.1)

        # 训练
        current_lr = scheduler.get_last_lr()[0]
        avg_loss, ap_meter = train(train_loader, model, criterion, optimizer, epoch,
                                   scaler, scheduler, warmup_scheduler, mixup_fn)

        if ema is not None:
            ema.update(model)

        print("Evaling......")
        train_ap, train_auroc, train_precision, train_recall, train_f1 = ap_meter.value()
        val_ap, val_auroc, val_precision, val_recall, val_f1, val_loss = evaluate(val_loader, model, criterion)

        print(
            f"Epoch {epoch + 1}, Train mAP: {train_ap:.4f}, ROC: {train_auroc:.4f}, train:loss:{avg_loss:.4e}; Validation mAP: {val_ap:.4f}, ROC:{val_auroc:.4f}, val_loss:{val_loss:.4e}")

        # 将mAP保存到日志文件
        log_file.write(
            f"Epoch {epoch + 1}, lr:{current_lr:.3e}, train_mAP: {train_ap:.4f}, train_loss:{avg_loss:.3e}, val_mAP: {val_ap:.4f}, val_precision:{val_precision:.4f}, val_recall:{val_recall:.4f}, val_f1_score:{val_f1:.4f}, val_ROC:{val_auroc:.4f}, val_loss:{val_loss:.4e}\n")
        log_file.flush()  # 强制刷新缓存

        # log_to_tensorboard(writer, epoch + 1, avg_loss, train_ap, val_loss, val_ap, current_lr, val_precision, val_recall, val_f1)
        write_to_csv(csv_filename, epoch + 1, avg_loss, train_ap, current_lr, val_loss, val_ap, val_precision,
                     val_recall, val_f1)

        save_checkpoint(epoch, model, optimizer, val_ap, warmup_scheduler, scheduler)
    print(start_time)
    best_epoch_checkpoint = torch.load(base_dir + f'{model_name}_best.pth')
    model = best_epoch_checkpoint['model']
    log_file.write('-' * 10 + 'val' + '-' * 10 + '\n')
    evaluate(val_loader, model, criterion, log_file, is_test=False)
    log_file.write('-' * 10 + 'test' + '-' * 10 + '\n')
    evaluate(test_loader, model, criterion, log_file, is_test=True)

    # 训练结束后关闭日志文件
    log_file.close()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler, warmup_scheduler, mixup_fn=None):
    model.train()

    Loss = []
    ap_meter = AveragePrecisionMeter2()
    ap_meter.reset()

    # 使用 tqdm 显示训练进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} Training",
                ncols=100)

    for i, (img_ols, img_sds, gt_s, flag) in pbar:
        img_ols = img_ols.to(device)
        img_sds = img_sds.to(device)
        gt_s = gt_s.to(device)
        flag = flag.to(device)
        if mixup:
            img_ols, gt_s1 = mixup_fn(img_ols, gt_s)
            img_sds, gt_s2 = mixup_fn(img_sds, gt_s)
            # print(f"\n gt_s_ori:{gt_s.shape}, gt_s_new {gt_s1.shape} img_ols_new {img_ols.shape}")
            # gt_s = gt_s1 + gt_s2

        if scaler is not None:
            # 使用 AMP 进行混合精度训练
            with torch.amp.autocast('cuda'):
                ol_output, sd_output = model(img_ols, img_sds)
                loss = criterion(ol_output, sd_output, gt_s, flag, alpha)
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            ol_output, sd_output = model(img_ols, img_sds)
            loss = criterion(ol_output, sd_output, gt_s)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # 记录损失
        cur_loss = loss.item()
        Loss.append(cur_loss)
        avg_loss = sum(Loss) / len(Loss)
        # 调整学习率
        # 评价训练集精度
        with torch.no_grad():
            prediction = torch.sigmoid(ol_output)
            # prediction = (prediction >= threshold).float()

            ap_meter.add(prediction.data, gt_s)
        cur_lr = learning_rate
        warmup_scheduler.step()  # 更新 warmup
        if scheduler is not None:
            cur_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'lr': f'{cur_lr:.3e}'})
    if scheduler is not None:
        scheduler.step()

    return avg_loss, ap_meter


def evaluate(val_loader, model, criterion, log_file=None, is_test=False):
    model.eval()  # 评估模式，不计算梯度

    ap_meter0 = AveragePrecisionMeter()
    ap_meter0.reset()

    ap_meter = AveragePrecisionMeter2()
    ap_meter.reset()

    Loss = []
    # 评估时禁用梯度计算
    with torch.no_grad():
        for i, (img_ols, img_sds, gt_s, flag) in enumerate(val_loader):
            img_ols = img_ols.to(device)
            img_sds = img_sds.to(device)
            gt_s = gt_s.to(device)

            # 模型预测
            main_output, ass_output = model(img_ols, img_sds)
            prediction = torch.sigmoid(main_output)
            # prediction = (prediction >= threshold).float()

            if only_eval or log_file is not None:
                ap_meter0.add(prediction.data, gt_s)
            ap_meter.add(prediction.data, gt_s)

            loss = criterion(main_output, ass_output, gt_s)
            Loss.append(loss.item())  # 记录放大后的损失

    avg_loss = sum(Loss) / len(Loss)

    ap, auroc, precision, recall, f1 = ap_meter.value()
    # 计算平均精度 (mAP)
    if only_eval or log_file is not None:
        if is_test:
            print(
                f"test_mAP: {ap}, test_precision:{precision}, test_recall:{recall}, test_f1_score:{f1}, test_ROC:{auroc}")
        else:
            print(f"val_mAP: {ap}, val_precision:{precision}, val_recall:{recall}, val_f1_score:{f1}, val_ROC:{auroc}")
        each_ap = ap_meter0.value()

        if log_file is not None:
            log_file.write(f"mAP: {ap}, precision:{precision}, recall:{recall}, f1_score:{f1}, ROC:{auroc}\n")
            for item_ap in each_ap:
                log_file.write(f"{item_ap} ")
            log_file.write(f"\n")
        print(each_ap)
        print(f"AP for each class: {each_ap.mean()}")

    return ap, auroc, precision, recall, f1, avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate Dual-view Model")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Only evaluate the model (default: False)")
    parser.add_argument("-r", "--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: None)")
    return parser.parse_args()


if __name__ == '__main__':
    main()
