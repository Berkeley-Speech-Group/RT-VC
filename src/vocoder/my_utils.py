import torch
from torch import nn
import torch.nn.functional as F

def calc_nparam(model):
    nparam = 0
    for p in model.parameters():
        if p.requires_grad:
            nparam += p.numel()
    return nparam

class ResBlock(nn.Module):
    '''
    Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
    '''
    def __init__(self, num_ins, num_outs, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)
    
class ConvNet(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=3, num_blocks = 2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, d_model, kernel_size, padding=(kernel_size-1)//2),
            *[ResBlock(d_model, d_model, kernel_size, padding=(kernel_size-1)//2) for _ in range(num_blocks-1)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, use_realtime=False, **kwargs):
        super().__init__(*args, **kwargs)
        if use_realtime:
            self.causal_padding = 0
        else:
            self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, use_realtime=False):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, use_realtime=use_realtime),
            nn.BatchNorm1d(out_channels),          
            nn.ReLU(),
        )
    
    def forward(self, x):
        return x + self.layers(x)

    def realtime_forward(self, x):
        # x in shape [1, in_channels, t], WITH context
        res = x
        temp = self.layers(x)
        context_len = res.shape[-1] - temp.shape[-1]
        res = res[:, :, context_len:]
        return res + temp

class CausalConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, use_realtime=False):
        super().__init__()
        self.layers = nn.Sequential(
            CausalResidualUnit(in_channels=hidden_dim, out_channels=hidden_dim, dilation=1, kernel_size=kernel_size, use_realtime=use_realtime),
            CausalResidualUnit(in_channels=hidden_dim, out_channels=hidden_dim, dilation=3, kernel_size=kernel_size, use_realtime=use_realtime),
            CausalResidualUnit(in_channels=hidden_dim, out_channels=hidden_dim, dilation=9, kernel_size=kernel_size, use_realtime=use_realtime),
        )

    def forward(self, x):
        return self.layers(x)
    
    def realtime_forward(self, x):
        for layer in self.layers:
            x = layer.realtime_forward(x)
        return x

class CausalConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, n_blocks, use_realtime=False):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            nn.BatchNorm1d(hidden_dim),          
            nn.ReLU(),
            *[CausalConvBlock(hidden_dim=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime) for _ in range(n_blocks)],
            CausalConv1d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=kernel_size, use_realtime=use_realtime)
        )
    def forward(self, x):
        return self.conv_blocks(x)
    
    def realtime_forward(self, x):
        for layer in self.conv_blocks:
            if isinstance(layer, CausalConvBlock):
                x = layer.realtime_forward(x)
            else:
                x = layer(x)
        return x

class DilatedConvStack(nn.Module):
    def __init__(self, hidden_dim, kernel_size, stride, dilations, use_transposed_conv, up_kernel_size):
        super().__init__()
        self.stack = []
        for dilation in dilations:
            self.stack.append(ResBlock(num_ins=hidden_dim, num_outs=hidden_dim, kernel_size=kernel_size, stride=stride, dilation=dilation))
        if use_transposed_conv:
            self.stack.extend([
                nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=up_kernel_size, stride=up_kernel_size//2, padding=up_kernel_size//4),
                nn.ReLU()
            ])
        self.stack = nn.Sequential(*self.stack)
        
    def forward(self, x):
        return self.stack(x)

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilations, nstacks, use_transposed_conv=False, up_kernel_size=None):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.stacks = []
        for _ in range(nstacks):
            self.stacks.append(DilatedConvStack(hidden_dim=out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, use_transposed_conv=use_transposed_conv, up_kernel_size=up_kernel_size))
        self.stacks = nn.Sequential(*self.stacks)
        self.out_conv = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.stacks(x)
        out = self.out_conv(x)
        return out

        
