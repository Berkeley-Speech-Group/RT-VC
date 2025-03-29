import torch
from torch import nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from torchaudio.transforms import MFCC

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, use_realtime=False, **kwargs):
        super().__init__(*args, **kwargs)
        if use_realtime:
            self.causal_padding = 0
        else:
            self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class ResidualUnit(nn.Module):
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

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, out_channels, kernel_size, use_realtime=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels,
                         out_channels=out_channels, dilation=1, kernel_size=kernel_size, use_realtime=use_realtime),
            ResidualUnit(in_channels=out_channels,
                         out_channels=out_channels, dilation=3, kernel_size=kernel_size, use_realtime=use_realtime),
            ResidualUnit(in_channels=out_channels,
                         out_channels=out_channels, dilation=9, kernel_size=kernel_size, use_realtime=use_realtime),
        )

    def forward(self, x):
        return self.layers(x)
    
    def realtime_forward(self, x):
        for layer in self.layers:
            x = layer.realtime_forward(x)
        return x
        
class InverterCausalConv(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, use_realtime=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.pre_conv = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            CausalConv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime)
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )      

    def forward(self, mfcc):
        """
        #### input ####
        mfcc     : the mfcc, in shape [B, t, in_channels]        
        #### output ####
        ema     : the predicted ema, in shape [B, t, 12]
        
        """
        output = self.pre_conv(mfcc.transpose(1,2)).transpose(1,2)
        ema = self.out_mlp(output)  # [B, t, 12]
        return ema

    @torch.no_grad()
    def infer_from_file(self, filepath, device):
        x, fs = sf.read(filepath)
        if fs != 16000:
            x = librosa.resample(x, orig_sr=fs, target_sr=16000)
            
        mfcc_transform = MFCC(
            sample_rate=16000,
            n_mfcc=24,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 128,
                "hop_length": 80,
                "mel_scale": "htk",
            },
        ).to(device)
        
        x = torch.from_numpy(x).to(device).float().unsqueeze(0)
        mfcc = mfcc_transform(x)
        ema = self.forward(mfcc.transpose(1, 2))
        return ema.transpose(-2, -1) # [1, 12, t]
        
    def realtime_forward(self, mfcc, mfcc_left_context):
        """chunk size is t
        #### input ####
        mfcc                : the mfcc, in shape [B, t, in_channels]  
        mfcc_left_context   : mfcc left context, in shape [B, CONTEXT_LEN, in_channels]
              
        #### output ####
        ema                 : the predicted ema, in shape [B, t, 12]
        
        """
        feat = torch.concat([mfcc_left_context, mfcc], dim=1).transpose(1,2)
        for layer in self.pre_conv:
            if isinstance(layer, EncoderBlock):
                feat = layer.realtime_forward(feat)
            else:
                feat = layer(feat)
        ema = self.out_mlp(feat.transpose(1,2))
        return ema
        