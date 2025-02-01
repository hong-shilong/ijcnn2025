import torch
import torch.nn as nn
import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, reduction='mean', scale_factor=1.0):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor  # 缩放因子，用于调整损失范围
        self.MLSloss = nn.MultiLabelSoftMarginLoss(reduction='none')

    def forward(self, main_output, ass_output, gt_s, flag=None, alpha=0.7):
        # 损失计算
        main_loss = self.MLSloss(main_output, gt_s)
        if ass_output is None or ass_output == 0 or flag is None:
            return main_loss.mean()

        # 获取beta及b0,b1
        beta = 1 - alpha
        b0 = beta // 2
        b1 = beta // 2

        # 扩散alpha, b0, b1
        alpha_expanded = alpha * torch.ones_like(main_loss)
        b0_expanded = b0 * torch.ones_like(main_loss)
        b1_expanded = b1 * torch.ones_like(main_loss)

        # 根据flag调整加权系数
        main_loss_weighted = main_loss * (alpha_expanded + (b0_expanded * (1-flag[:, 0])) + (b1_expanded * (1-flag[:, 1])))

        # 辅助损失计算
        ol_ass_loss = self.MLSloss(ass_output[0], gt_s)
        sd_ass_loss = self.MLSloss(ass_output[1], gt_s)

        # 对辅助损失进行加权
        ass_loss_weighted = (ol_ass_loss * (b0_expanded * flag[:, 0])) + (sd_ass_loss * (b1_expanded * flag[:, 1]))

        # 合并损失
        total_loss = main_loss_weighted + ass_loss_weighted

        # 计算最终损失
        if self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'mean':
            return total_loss.mean()
        else:
            return total_loss

