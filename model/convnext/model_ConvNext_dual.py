import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class MyModel(nn.Module):

    def __init__(self, num_classes=15, depth_mult=1, expansion=1/4, act="silu", dropout=0.1):

        super(MyModel, self).__init__()

        backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        backbone = backbone.features
        # print(backbone)
        for item in backbone.children():
            if isinstance(item, nn.BatchNorm2d):
                item.affine = False
        # 初始特征层
        self.features = nn.Sequential(
        )

        # 将每个 block 命名为类似 ResNet 的命名方式
        self.layer1 = backbone[:2]
        self.layer2 = backbone[2:4]
        self.layer3 = backbone[4:6]
        self.layer4 = backbone[6:]
        output_channels = [96, 192, 384, 768]

        self.cov4 = nn.Conv2d(output_channels[3]*2, output_channels[3]*2, kernel_size=1, stride=1)

        self.po1 = nn.AvgPool2d(8, stride=1)
        self.po2 = nn.AvgPool2d(16, stride=1)
        self.po3 = nn.AvgPool2d(32, stride=1)

        self.fc = nn.Linear(output_channels[3]*2, num_classes)

    def forward(self, image_ol, image_sd):

        bf_ol = self.features(image_ol)
        f_l1_o1 = self.layer1(bf_ol)
        f_l2_ol = self.layer2(f_l1_o1)
        f_l3_ol = self.layer3(f_l2_ol)
        f_l4_ol = self.layer4(f_l3_ol)

        bf_sd = self.features(image_sd)
        f_l1_sd = self.layer1(bf_sd)
        f_l2_sd = self.layer2(f_l1_sd)
        f_l3_sd = self.layer3(f_l2_sd)
        f_l4_sd = self.layer4(f_l3_sd)
        out = torch.cat((f_l4_ol, f_l4_sd), dim=1).contiguous()

        out = self.fc(self.po1(F.relu(self.cov4(out))).view(out.size(0), -1))
        # out1_sd = self.fc1_sd(self.po1(F.relu(self.cov4_sd(f_l4_sd))).view(f_l4_sd.size(0), -1))
        
        ol_output = out

        # sd_output = out1_sd

        return ol_output, 0

if __name__ == '__main__':

    from thop import profile
    from thop import clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(num_classes=15).to(device)

    flops, params = profile(model, (torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device)))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)


