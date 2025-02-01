import torch
from torch import nn
from torchvision import models

from module.CBAM import CBAM
from module.DynamicFilter import DynamicFilter
from module.csp import CSPRepLayer, DWLayer


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048):
        """
        多头互注意力模块，支持位置编码
        :param d_model: 特征维度
        :param nhead: 多头数量
        :param dropout: dropout 概率
        """
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # 位置编码相关
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        """
        添加位置编码
        :param tensor: 输入特征 [B, N, C]
        :param pos_embed: 位置编码 [B, N, C]
        """
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x1, x2, pos1=None, pos2=None):
        """
        前向传播
        :param x1: 输入 1（Query），形状 [B, N1, C]
        :param x2: 输入 2（Key, Value），形状 [B, N2, C]
        :param pos1: 输入 1 的位置编码，形状 [B, N1, C]
        :param pos2: 输入 2 的位置编码，形状 [B, N2, C]
        :return: 输出特征，形状 [B, N1, C]
        """

        residual = x2
        
        q = self.with_pos_embed(x1, pos1)
        k = self.with_pos_embed(x2, pos2)
        v = x2
        # 使用 x1 作为 Query，x2 作为 Key 和 Value
        x2, _ = self.self_attn(q, k, v)
        x2 = residual + self.dropout(x2)
        src = self.norm(x2)
        return src


class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, temperature=10000.):
        """
        2D 正余弦位置编码
        :param embed_dim: 编码维度
        :param temperature: 控制编码频率的温度参数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, h, w, device='cpu'):
        """
        生成位置编码
        :return: 位置编码，形状 [1, H*W, C]
        """
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1. / (self.temperature ** omega)

        out_w = torch.einsum('i,j->ij', grid_w.flatten(), omega)  # [H*W, pos_dim]
        out_h = torch.einsum('i,j->ij', grid_h.flatten(), omega)

        pos_embed = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
        return pos_embed.unsqueeze(0)  # [1, H*W, C]

class MultiScaleSightFeatureEnhance(nn.Module):
    def __init__(self, ch_in, ch_next=None, num_heads=8):
        super(MultiScaleSightFeatureEnhance, self).__init__()
        # 多头互注意力模块
        self.cross_attention = MultiHeadCrossAttention(d_model=ch_in, nhead=num_heads)
        # self.cross_attention = TKSA(ch_in, num_heads, None)
        self.norm = nn.BatchNorm2d(ch_in)
        # self.norm2 = nn.BatchNorm2d(ch_in)
        # 位置编码器
        self.pos_encoder_ol = SinCosPositionalEncoding2D(
            embed_dim=ch_in
        )
        self.pos_encoder_sd = SinCosPositionalEncoding2D(
            embed_dim=ch_in
        )
        self.ch_next = ch_next
        if self.ch_next is not None:
            # self.ffn = DownLayer(ch_in, ch_next)
            self.ffn1 = DWLayer(ch_in, ch_next)
            self.ffn2 = DWLayer(ch_in, ch_next)
            self.spatial = nn.Conv2d(2, 1, 7, stride=1, padding=(7 - 1) // 2)

    def forward(self, f_ol, f_sd):
        """
        :param h_x: 高分辨率输入特征 [B, C, H1, W1]
        :param l_x: 低分辨率输入特征 [B, C, H2, W2]
        """
        B, C, H, W = f_ol.size()

        # 将特征展平为 [B, N, C]
        f_ol_flat = f_ol.flatten(2).permute(0, 2, 1)  # [B, H1*W1, C]
        f_sd_flat = f_sd.flatten(2).permute(0, 2, 1)  # [B, H2*W2, C]
        # f_ol_flat = f_ol
        # f_sd_flat = f_sd
        # 生成位置编码
        f_ol_flat = self.pos_encoder_ol(h=H, w=W, device=f_ol.device)+f_ol_flat  # 更新后的 H1, W1
        f_sd_flat = self.pos_encoder_sd(h=H, w=W, device=f_ol.device)+f_sd_flat

        # 应用互注意力
        f_sd_attended = self.cross_attention(f_ol_flat, f_sd_flat) 
        f_ol_attended = self.cross_attention(f_sd_flat, f_ol_flat) 


        # 恢复特征维度
        f_ol_attended = f_ol_attended.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H2, W2]
        f_sd_attended = f_sd_attended.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H2, W2]
        
        f_ol = self.norm(f_ol_attended) + f_ol
        f_sd = self.norm(f_sd_attended) + f_sd
        
        s_ol = s_sd = 1
        if self.ch_next is not None:
            ffn_ol = self.ffn1(f_ol)
            ffn_sd = self.ffn2(f_sd)
    
            # # 计算平均池化和最大池化，保持维度 [B, 1, H, W]
            # avg_ol = torch.mean(ffn_ol, dim=1, keepdim=True)  # [B, 1, H, W]
            # max_ol, _ = torch.max(ffn_ol, dim=1, keepdim=True)  # [B, 1, H, W]
            # avg_sd = torch.mean(ffn_sd, dim=1, keepdim=True)  # [B, 1, H, W]
            # max_sd, _ = torch.max(ffn_sd, dim=1, keepdim=True)  # [B, 1, H, W]
            
            # # 合并池化结果
            s_ol = torch.cat((torch.max(ffn_ol, 1)[0].unsqueeze(1), torch.mean(ffn_ol, 1).unsqueeze(1)), dim=1)
            s_ol = self.spatial(s_ol) # [B, 1, H, W]
            
            s_sd = torch.cat((torch.max(ffn_sd, 1)[0].unsqueeze(1), torch.mean(ffn_sd, 1).unsqueeze(1)), dim=1)
            s_sd = self.spatial(s_sd)  # [B, 1, H, W]
            
            # 激活函数
            s_ol = torch.sigmoid(ffn_ol)  # [B, 1, H, W]
            s_sd = torch.sigmoid(ffn_sd)  # [B, 1, H, W]
        return f_ol, f_sd, s_ol, s_sd

# Phase integration module
class CrossSightFreqInteract(nn.Module):
    def __init__(self, channel):
        super(CrossSightFreqInteract, self).__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel*2, channel*2, 1, 1, 0),
            nn.Sigmoid()
        )

        self.processpha = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel*2, channel*2, 1, 1, 0),
            nn.Sigmoid()
        )
        self.freq_ehance = DynamicFilter(channel, size=64)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x, y):
        _, C, H, W = x.shape

        x = self.freq_ehance(x) + x
        y = self.freq_ehance(y) + y
        return x, y
        # with torch.cuda.amp.autocast(dtype=torch.float32):
    
        x_fft = torch.fft.rfft2(x, norm='ortho')
        y_fft = torch.fft.rfft2(y, norm='ortho')
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        
        y_amp = torch.abs(y_fft)
        y_phase = torch.angle(y_fft)

        # mix_phase = torch.cat([x_phase, y_phase], dim=1)
        # mix_phase = self.processpha(mix_phase)
        # x_phase = mix_phase[:, :C, :, :]*x_phase + x_phase
        # y_phase = mix_phase[:, C:, :, :]*y_phase + y_phase

        # mix_amp = torch.cat([x_amp, y_amp], dim=1)
        # mix_amp = self.processmag(mix_amp)
        # x_amp = mix_amp[:, :C, :, :]*x_amp + x_amp
        # y_amp = mix_amp[:, C:, :, :]*y_amp + y_amp

        # 重建频域特征l
        out_x = x_amp * torch.exp(1j * x_phase)
        out_y = y_amp * torch.exp(1j * y_phase)  
                
        out_x = torch.fft.irfft2(out_x, s=(H, W), norm='ortho')
        out_y = torch.fft.irfft2(out_y, s=(H, W), norm='ortho')
        
        # out_x = self.bn(x)
        # out_y = self.bn(y)

        return out_x, out_y

class MutilSightFuseModule(nn.Module):
    def __init__(self, int_c, out_c, depth_mult=1, act='silu', expansion=1):
        super(MutilSightFuseModule, self).__init__()
        self.attn = CBAM(int_c//2)     
        self.bn1 = nn.BatchNorm2d(int_c)
        self.csp = CSPRepLayer(int_c, out_c, round(1 * depth_mult), act=act, expansion=expansion, use_Dcn=False, use_wt=False)
        self.act = nn.ReLU(inplace=True)  # 经典ReLU
        self.bn2 = nn.BatchNorm2d(int_c)


    def forward(self, h_x, l_x):
        # res = torch.cat((h_x, l_x), dim=1).contiguous()
        # return res

        h_x = self.attn(h_x)
        l_x = self.attn(l_x)
        res = torch.cat((h_x, l_x), dim=1).contiguous()
        res = self.bn1(res)
        res = self.csp(res)
        # res = self.bn2(self.act(res))
        return res
        
class MyModel(nn.Module):

    def __init__(self, num_classes=15, depth_mult=3, expansion=1/4, act="silu", dropout=0.1):

        super(MyModel, self).__init__()

        backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        backbone = backbone.features
        # print(backbone)
        #for item in backbone.children():
            #if isinstance(item, nn.BatchNorm2d):
                #item.affine = False
        # 初始特征层
        self.features = nn.Sequential(
            backbone[0],
        )

        # 将每个 block 命名为类似 ResNet 的命名方式
        self.layer1 = backbone[1]
        self.layer2 = backbone[2:4]
        self.layer3 = backbone[4:6]
        self.layer4 = backbone[6:]
        output_channels = [96, 192, 384, 768]
        
        self.freq_fuse = CrossSightFreqInteract(output_channels[0])
        # self.cfe1 = MultiScaleSightFeatureEnhance(256)
        self.cfe2 = MultiScaleSightFeatureEnhance(output_channels[1], output_channels[2])
        self.cfe3 = MultiScaleSightFeatureEnhance(output_channels[2], output_channels[3])
        self.cfe4 = MultiScaleSightFeatureEnhance(output_channels[3])
        output_dim = output_channels[3]
        self.ol_sd_fuse = MutilSightFuseModule(output_channels[3]*2, 
                                               output_dim, 
                                               depth_mult=depth_mult
                                              )
        # output_dim = 4096
        # print(output_dim)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(8, stride=1),
            nn.Flatten(),
            nn.Linear(output_dim, output_dim//2),
            nn.Dropout(dropout),
            nn.Linear(output_dim//2, num_classes)
            # nn.Linear(output_dim, num_classes)

        )

    def forward(self, image_ol, image_sd):
        bf_ol = self.features(image_ol)
        bf_sd = self.features(image_sd)

        f_l1_ol = self.layer1(bf_ol)
        f_l1_sd = self.layer1(bf_sd)
        # with torch.amp.autocast('cuda', enabled=False):  
        # print(f_l1_ol.shape)
        f_l1_ol, f_l1_sd = self.freq_fuse(f_l1_ol, f_l1_sd)
        
        f_l2_ol = self.layer2(f_l1_ol)
        f_l2_sd = self.layer2(f_l1_sd)
        f_l2_ol, f_l2_sd, s_ol, s_sd = self.cfe2(f_l2_ol, f_l2_sd)


        f_l3_ol = self.layer3(f_l2_ol)
        f_l3_sd = self.layer3(f_l2_sd)
        f_l3_ol, f_l3_sd, s_ol, s_sd = self.cfe3(f_l3_ol*s_ol+f_l3_ol, f_l3_sd+f_l3_sd*s_sd)

        f_l4_ol = self.layer4(f_l3_ol)
        f_l4_sd = self.layer4(f_l3_sd)
        f_l4_ol, f_l4_sd, _, _ = self.cfe4(f_l4_ol*s_ol+f_l4_ol, f_l4_sd+f_l4_sd*s_sd)
        # Output all shapes in one print
        out = self.ol_sd_fuse(f_l4_ol, f_l4_sd)
        # print(out.shape)
        out = self.classifier(out)

        return out, 0

if __name__ == "__main__":

    from thop import profile
    from thop import clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(num_classes=15).to(device)

    flops, params = profile(model, (torch.randn(1, 3, 256, 256).to(device), torch.randn(1, 3, 256, 256).to(device)))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
