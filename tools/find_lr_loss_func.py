import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_WithLogitsLoss = nn.BCEWithLogitsLoss()
        # BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_out, y_true):
        a =  self.bce_WithLogitsLoss(y_out, y_true)
        b =  self.soft_dice_loss(y_pred, y_true)
        return (0.5 * a + 0.5 * b) * 2


class BCELoss(nn.Module):
    def __init__(self, reduction='mean', scale_factor=1.0):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor  # 缩放因子，用于调整损失范围
        self.Loss = nn.BCEWithLogitsLoss(reduction='none')  # 基础 BCE 损失
        # self.loss_fct_asy = AsymmetricLoss()  # Asymmetric Loss
        # self.MLSloss = nn.MultiLabelSoftMarginLoss()
    def forward(self, x, gt_s, a=1, epsilon=1e-8):
        # OL 分支损失
        loss = self.Loss(x, gt_s)

        # 按任务权重 a 平衡 OL 和 SD 任务的重要性
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
