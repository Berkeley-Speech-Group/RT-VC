import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

def f2p(f, f_base):
    p = 12 * np.log2(f / f_base)
    return p

def p2f(p, f_base):
    f = f_base * 2 ** (p / 12)
    return f

def quantize(f, nbins, fmin, fmax, f_base):
    pmin = f2p(fmin, f_base)
    pmax = f2p(fmax, f_base)
    delta_p = (pmax - pmin) / (nbins - 1)
    
    # clamp f into the range of [fmin, fmax]
    f = np.clip(f, fmin, fmax)
    
    # quantize f into bin indices
    p = f2p(f, f_base)
    idx = (np.round((p - pmin) / delta_p)).astype(int) # should be an int in the range of [0, nbins-1]
    return idx

def dequantize(idx, nbins, fmin, fmax, f_base):
    pmin = f2p(fmin, f_base)
    pmax = f2p(fmax, f_base)
    delta_p = (pmax - pmin) / (nbins - 1)
    p_q = pmin + idx * delta_p
    
    return p2f(p_q, f_base)

class MLP(nn.Module):
    def __init__(self, input_channel, hidden_dim, output_channel, forward_expansion=1, n_midlayers=1, dropout=0):
        super().__init__()
        layers = [
            nn.Linear(input_channel, hidden_dim*forward_expansion),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_midlayers):
            layers.extend([
                nn.Linear(hidden_dim*forward_expansion, hidden_dim*forward_expansion),
                nn.ReLU()
            ])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim*forward_expansion, output_channel),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        #### input ####
        x   : in shape [*, input_channel]
        
        #### output ####
        out : in shape [*, output_channel]
        
        """
        out = self.layers(x)
        return out

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

class CausalConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, use_realtime=False):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            EncoderBlock(out_channels=hidden_dim, kernel_size=kernel_size, use_realtime=use_realtime),
            CausalConv1d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=kernel_size, use_realtime=use_realtime)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def realtime_forward(self, x):
        for layer in self.layers:
            if isinstance(layer, EncoderBlock):
                x = layer.realtime_forward(x)
            else:
                x = layer(x)
        return x
        
class RTPitchTracker(nn.Module):
    def __init__(self, config, use_realtime=False):
        super().__init__()
        self.config = config
        self.conv = CausalConvNet(**config['convnet_config'], use_realtime=use_realtime)
        self.pitch_head = MLP(
            input_channel=config['hidden_dim'],
            hidden_dim=config['hidden_dim'],
            output_channel=config['nbins'], 
            forward_expansion=config['forward_expansion'],
            n_midlayers=config['n_midlayers'],
        )
        
        if config['use_periodicity_head']:
            self.periodicity_head = nn.Sequential(
                MLP(
                    input_channel=config['hidden_dim'],
                    hidden_dim=config['hidden_dim'],
                    output_channel=1, 
                    forward_expansion=config['forward_expansion'],
                    n_midlayers=config['n_midlayers'],
                ),
                nn.Sigmoid()
            )
        else:
            self.periodicity_head = None
            
        if config['use_loudness_head']:
            self.loudness_head = nn.Sequential(
                MLP(
                    input_channel=config['hidden_dim'],
                    hidden_dim=config['hidden_dim'],
                    output_channel=1, 
                    forward_expansion=config['forward_expansion'],
                    n_midlayers=config['n_midlayers'],
                ),
                nn.Sigmoid()
            )
            self.scale_factor = 8
        else:
            self.loudness_head = None
    
    def forward(self, mel):
        """
        #### input ####
        mel         : in shape [B, 128, t] @200Hz
        
        #### output ####
        logits      : in shape [B, t, nbins] @200Hz
        periodicity : in shape [B, 1, t] @200Hz
        loudness    : in shape [B, 1, t] @200Hz
        
        """
        temp = self.conv(mel).transpose(1, 2) # [B, t, hidden_dim]
        logits = self.pitch_head(temp) # [B, t, nbins]
        
        if self.periodicity_head:
            periodicity = self.periodicity_head(temp).transpose(1,2) # [B, 1, t]
        else:
            periodicity = None
            
        if self.loudness_head:
            loudness = self.scale_factor * self.loudness_head(temp).transpose(1,2) # [B, 1, t]
        else:
            loudness = None
            
        return {
            'logits': logits,
            'periodicity': periodicity,
            'loudness': loudness
        }
    
    def infer(self, mel):
        """
        #### input ####
        mel         : in shape [B, 128, t] @200Hz
        
        #### output ####
        pitch       : in shape [B, 1, t] @200Hz
        periodicity : in shape [B, 1, t] @200Hz
        loudness    : in shape [B, 1, t] @200Hz
        
        """
        out_dict = self.forward(mel)
        logits = out_dict['logits'] # [B, t, nbins]
        probs = torch.sigmoid(logits) # [B, t, nbins]
        periodicity = out_dict['periodicity'] # [B, 1, t]
        loudness = out_dict['loudness'] # [B, 1, t]
        
        weighted_sum = []
        L = 9
        for b in range (probs.shape[0]):
            prob = probs[b] # [t, nbins]
            center = torch.argmax(prob, dim=-1) # [t, ]
            start = torch.clamp(center-4, min=0) # [t, ]
            indices = torch.arange(L, device=mel.device).unsqueeze(0) + start.unsqueeze(1)
            indices = torch.clamp(indices, max=self.config['nbins']-1) # [t, L]
            seg_prob = prob[torch.arange(prob.shape[0], device=mel.device).unsqueeze(1), indices] # [t, L]
            s = ((indices * seg_prob).sum(dim=-1) / seg_prob.sum(dim=-1)).unsqueeze(0) # [1, t]
            weighted_sum.append(s)
        
        weighted_sum = torch.stack(weighted_sum, dim=0) # [B, 1, t]
        pitch = dequantize(weighted_sum, self.config['nbins'], self.config['fmin'], self.config['fmax'], self.config['f_base']) # [B, 1, t]
        return {
            'pitch': pitch,
            'periodicity': periodicity,
            'loudness': loudness
        }
        
    @torch.no_grad()
    def infer_from_file(self, filepath, device):
        """        
        #### output ####
        pitch       : in shape [B, 1, t] @200Hz
        periodicity : in shape [B, 1, t] @200Hz
        loudness    : in shape [B, 1, t] @200Hz
        
        """
        x, fs = sf.read(filepath)
        if fs != 16000:
            x = librosa.resample(x, orig_sr=fs, target_sr=16000)
        x = torch.from_numpy(x).to(device).float()
        transform = MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=80, n_mels=128).to(device)
        mel = transform(x).unsqueeze(0)
        
        out_dict = self.forward(mel)
        logits = out_dict['logits'] # [B, t, nbins]
        probs = torch.sigmoid(logits) # [B, t, nbins]
        periodicity = out_dict['periodicity'] # [B, 1, t]
        loudness = out_dict['loudness'] # [B, 1, t]
        
        weighted_sum = []
        L = 9
        for b in range (probs.shape[0]):
            prob = probs[b] # [t, nbins]
            center = torch.argmax(prob, dim=-1) # [t, ]
            start = torch.clamp(center-4, min=0) # [t, ]
            indices = torch.arange(L, device=mel.device).unsqueeze(0) + start.unsqueeze(1)
            indices = torch.clamp(indices, max=self.config['nbins']-1) # [t, L]
            seg_prob = prob[torch.arange(prob.shape[0], device=mel.device).unsqueeze(1), indices] # [t, L]
            s = ((indices * seg_prob).sum(dim=-1) / seg_prob.sum(dim=-1)).unsqueeze(0) # [1, t]
            weighted_sum.append(s)
        
        weighted_sum = torch.stack(weighted_sum, dim=0) # [B, 1, t]
        pitch = dequantize(weighted_sum, self.config['nbins'], self.config['fmin'], self.config['fmax'], self.config['f_base']) # [B, 1, t]
        return {
            'pitch': pitch,
            'periodicity': periodicity,
            'loudness': loudness
        }