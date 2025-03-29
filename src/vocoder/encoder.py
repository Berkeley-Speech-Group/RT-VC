import torch
from torch import nn
import math
import torch.random
from transformers import WavLMModel
from vocoder.my_utils import DilatedConvEncoder, CausalConvNet

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

def freeze_module(module):
    for p in module.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
    module.eval()

class SpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.freeze_wavlm = config['freeze_wavlm']
        self.use_std = config['use_std']
        speech_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm_conv = speech_model.feature_extractor
        if self.freeze_wavlm:
            freeze_module(self.wavlm_conv)
        else:
            self.wavlm_conv.train()
        self.conv_stack = DilatedConvEncoder(in_channels=512, out_channels=config['hidden_dim'], kernel_size=3, stride=1, dilations=config['dilations'], nstacks=config['nstacks'])

        input_dim_factor = 2 if self.use_std else 1
        self.spkr_mlp = MLP(input_channel=config['hidden_dim']*input_dim_factor, hidden_dim=config['spkr_emb_dim'], output_channel=config['spkr_emb_dim'], forward_expansion=config['forward_expansion'], n_midlayers=config['n_midlayers'])
        
    def forward(self, x, periodicity):
        """
        #### input ####
        x           : in shape [B, t], sampled @16kHz
        periodicity : in shape [B, 1, t_model], sampled @200Hz
        
        #### output ####
        spkr_emb    : in shape [B, spkr_emb_dim]
        
        """
        # wavlm conv
        temp = self.wavlm_conv(x) # [B, 512, t'], @50Hz
        
        # conv stack
        temp = self.conv_stack(temp) # [B, 512, t'], @50Hz
        
        # attentive statistical pooling
        periodicity = periodicity[:, :, ::4] 
        min_len = min(periodicity.shape[-1], temp.shape[-1])
        temp = temp[:, :, :min_len] # [B, 512, t'], @50Hz
        periodicity = periodicity[:, :, :min_len] # [B, 1, t'], @50Hz
        weight = periodicity / (periodicity.sum(-1, keepdim=True) + 1e-4)
        
        vec = (weight * temp).sum(-1) # [B, 512]
        if self.use_std:
            std = torch.sqrt((weight * temp**2).sum(-1) - vec**2) # [B, 512]
            vec = torch.concat([vec, std], dim=-1) # [B, 1024]
        
        # mlp
        spkr_emb = self.spkr_mlp(vec) # [B, spkr_emb_dim]
        
        return spkr_emb

class LinearFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, spkr_emb_dim):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.film = nn.Linear(spkr_emb_dim, out_channels*2)
                
    def forward(self, x, spkr_emb):
        # x in shape [B, t, in_channels], spkr_emb in shape [B, spkr_emb_dim]
        x = self.linear(x) # [B, t, out_channels]
        condition = self.film(spkr_emb).unsqueeze(1) # [B, 1, out_channels*2] 
        out = condition[:, :, :self.out_channels] * x + condition[:, :, self.out_channels:]
        return out
        
class MLPFiLM(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, spkr_emb_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearFiLM(in_channels=in_channels, out_channels=hidden_dim, spkr_emb_dim=spkr_emb_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        ])
    
    def forward(self, x, spkr_emb):
        # x in shape [B, t, in_channels], spkr_emb in shape [B, spkr_emb_dim]
        out = x
        for layer in self.layers:
            if not isinstance(layer, LinearFiLM):
                out = layer(out)
            else:
                out = layer(out, spkr_emb) # [B, t, out_channels]
        return out
        
class EncoderCausalConvSinCos(nn.Module):
    def __init__(self, hidden_dim=256, nharmonics=80, nbands=257, attenuate=0.01, kernel_size=3, n_blocks=3, spkr_emb_dim=128, forward_expansion=4, use_realtime=False):
        super().__init__()
        self.nharmonics = nharmonics
        self.attenuate = attenuate
        
        self.causal_conv = CausalConvNet(in_channels=15, hidden_dim=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, n_blocks=n_blocks, use_realtime=use_realtime)

        self.head_amp = MLPFiLM(in_channels=hidden_dim, hidden_dim=hidden_dim*forward_expansion, out_channels=(nharmonics+1)*2, spkr_emb_dim=spkr_emb_dim)
        self.head_H = MLPFiLM(in_channels=hidden_dim, hidden_dim=hidden_dim*forward_expansion, out_channels=nbands, spkr_emb_dim=spkr_emb_dim)

    # Scale sigmoid as per original DDSP paper
    def _scaled_sigmoid(self, x):
        return 2.0 * (torch.sigmoid(x) ** math.log(10)) + 1e-7

    def forward(self, f0, loudness, periodicity, ema, spkr_emb):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model
        loudness    : in shape [B, 1, t], calculated at f_model
        periodicity : in shape [B, 1, t], calculated at f_model
        ema         : in shape [B, 12, t], sampled at f_model
        spkr_emb    : in shape [B, spkr_emb_dim]

        #### output ####
        cn          : in shape [B, nharmonics, t], calculated at f_model
        an          : in shape [B, 1, t], calculated at f_model
        H           : in shape [B, t, nbands], calculated at f_model

        """
        # normalize f0 to be within [0, 1]
        f0 = f0 / 500

        # conv mapping
        in_feat = torch.concat([f0, loudness, periodicity, ema], dim=1) # [B, 15, t]
        out_feat = self.causal_conv(in_feat).transpose(-2, -1)  # [B, t, hidden_dim]

        # output heads
        amp = self.head_amp(out_feat, spkr_emb)  # [B, t, (nharmonics+1)*2]
        amp_sin = amp[:, :, :(self.nharmonics + 1)]  # [B, t, nharmonics+1]
        amp_cos = amp[:, :, (self.nharmonics + 1):]  # [B, t, nharmonics+1]

        cn_sin = (amp_sin[:, :, 1:]).transpose(1, 2)  # [B, nharmonics, t]
        an_sin = self._scaled_sigmoid(amp_sin[:, :, 0].unsqueeze(-1)).transpose(1, 2) # [B, 1, t]

        cn_cos = (amp_cos[:, :, 1:]).transpose(1, 2)  # [B, nharmonics, t]
        an_cos = self._scaled_sigmoid(amp_cos[:, :, 0].unsqueeze(-1)).transpose(1, 2) # [B, 1, t]

        H = self._scaled_sigmoid(self.head_H(out_feat, spkr_emb)) * self.attenuate

        return cn_sin, cn_cos, an_sin, an_cos, H
    
    def realtime_forward(self, f0, loudness, periodicity, ema, spkr_emb):
        """WITH context!
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model
        loudness    : in shape [B, 1, t], calculated at f_model
        periodicity : in shape [B, 1, t], calculated at f_model
        ema         : in shape [B, 12, t], sampled at f_model
        spkr_emb    : in shape [B, spkr_emb_dim]

        #### output ####
        cn          : in shape [B, nharmonics, t], calculated at f_model
        an          : in shape [B, 1, t], calculated at f_model
        H           : in shape [B, t, nbands], calculated at f_model

        """
        # normalize f0 to be within [0, 1]
        f0 = f0 / 500

        # conv mapping
        in_feat = torch.concat([f0, loudness, periodicity, ema], dim=1) # [B, 15, t]
        out_feat = self.causal_conv.realtime_forward(in_feat).transpose(-2, -1)  # [B, t, hidden_dim]

        # output heads
        amp = self.head_amp(out_feat, spkr_emb)  # [B, t, (nharmonics+1)*2]
        amp_sin = amp[:, :, :(self.nharmonics + 1)]  # [B, t, nharmonics+1]
        amp_cos = amp[:, :, (self.nharmonics + 1):]  # [B, t, nharmonics+1]

        cn_sin = (amp_sin[:, :, 1:]).transpose(1, 2)  # [B, nharmonics, t]
        an_sin = self._scaled_sigmoid(amp_sin[:, :, 0].unsqueeze(-1)).transpose(1, 2) # [B, 1, t]

        cn_cos = (amp_cos[:, :, 1:]).transpose(1, 2)  # [B, nharmonics, t]
        an_cos = self._scaled_sigmoid(amp_cos[:, :, 0].unsqueeze(-1)).transpose(1, 2) # [B, 1, t]

        H = self._scaled_sigmoid(self.head_H(out_feat, spkr_emb)) * self.attenuate

        return cn_sin, cn_cos, an_sin, an_cos, H