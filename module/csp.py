import torch.nn as nn
from .utils import get_activation


class DWConv(nn.Module):
    """Depth-wise layer with optional pooling and activation."""
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=None, bias=False, act=None):
        """
        Initialize a Depth-wise layer.

        Args:
            ch_in (int): Number of input channels.
            ch_out (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
            padding (int, optional): Padding for the convolution. Auto-calculated if None.
            bias (bool): Whether to use bias in convolution. Default is False.
            act (str or nn.Module, optional): Activation function. Default is None (Identity).
            use_wt (bool): Placeholder for potential weight usage. Default is False.
        """
        super().__init__()
        # Auto padding if not specified
        padding = (kernel_size - 1) // 2 if padding is None else padding
        
        # Depth-wise convolution
        self.Dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                               groups=ch_in, bias=bias)
        # Point-wise convolution
        self.Wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        
        # Batch normalization
        self.norm = nn.BatchNorm2d(ch_out)
        
        # Activation function
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """
        Forward pass through the Depth-wise layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        x = self.Dconv(x)  # Depth-wise convolution
        x = self.Wconv(x)  # Point-wise convolution
        x = self.norm(x)   # Batch normalization
        x = self.act(x)    # Activation
        return x
        
class DWLayer(nn.Module):
    """Depth-wise layer with optional pooling and activation."""
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=None, bias=False, act=None):
        """
        Initialize a Depth-wise layer.

        Args:
            ch_in (int): Number of input channels.
            ch_out (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
            padding (int, optional): Padding for the convolution. Auto-calculated if None.
            bias (bool): Whether to use bias in convolution. Default is False.
            act (str or nn.Module, optional): Activation function. Default is None (Identity).
            use_wt (bool): Placeholder for potential weight usage. Default is False.
        """
        super().__init__()
        # Auto padding if not specified
        padding = (kernel_size - 1) // 2 if padding is None else padding
        
        # Depth-wise convolution
        self.Dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                               groups=ch_in, bias=bias)
        # Point-wise convolution
        self.Wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        
        # Batch normalization
        self.norm = nn.BatchNorm2d(ch_out)
        
        # Activation function
        self.act = nn.Identity() if act is None else get_activation(act)
        
        # Optional pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the Depth-wise layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        x = self.Dconv(x)  # Depth-wise convolution
        x = self.Wconv(x)  # Point-wise convolution
        x = self.norm(x)   # Batch normalization
        x = self.act(x)    # Activation
        x = self.pool(x)   # Pooling (if enabled)
        return x

class DownLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=1, stride=1, padding=None, bias=False, act=None, use_wt=False):
        super().__init__()
        # self.conv = DWConv(ch_in, ch_out, 3, stride)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.pool(x)
        return self.act(self.norm(self.conv(x)))
        
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None, use_wt=False, use_Dcn=False):
        super().__init__()
        if use_wt and ch_in==ch_out:
            self.conv = WTConv2d(ch_in, ch_out, kernel_size, stride, bias=bias)
        elif use_Dcn and kernel_size!=1:
            # self.conv = DWConv(ch_in, ch_out, kernel_size, stride)
            self.conv = StarBlock(ch_out)
        else:
            self.conv = nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2 if padding is None else padding,
                bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 useglu=False,
                 use_wt=False,
                 use_Dcn=False):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, 
                                   bias=bias, act=act, use_wt=use_wt, use_Dcn=use_Dcn)
        self.bottlenecks = DWConv(in_channels, hidden_channels)
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, 
                                       bias=bias, act=act, use_wt=use_wt, use_Dcn=use_Dcn)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        # x = self.conv1(x)
        x_1 = self.bottlenecks(x)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
