import torch
import torch.nn as nn
from torchaudio.functional import fftconvolve
from vocoder.encoder import EncoderCausalConvSinCos, MLP
from vocoder.harmonic import HarmonicOscillatorSinCos
from vocoder.noise import FilteredNoiseGenerator

class Vocoder(nn.Module):
    def __init__(self, hidden_dim, nharmonics, nbands, attenuate=0.01, fs=16000, framesize=80, 
                 spkr_emb_dim=128, kernel_size=3, n_blocks=3, forward_expansion=4, reverb_len=1025, use_realtime=False):
        super().__init__()
        self.reverb_len = reverb_len
        self.framesize = framesize
        
        self.encoder = EncoderCausalConvSinCos(
            hidden_dim, nharmonics, nbands, attenuate, 
            kernel_size, n_blocks, spkr_emb_dim, forward_expansion, use_realtime
        )
        self.harmonic = HarmonicOscillatorSinCos(fs, framesize)
        self.noise = FilteredNoiseGenerator(framesize, nbands)
        self.reverb_mlp = MLP(input_channel=spkr_emb_dim, hidden_dim=reverb_len, output_channel=reverb_len, forward_expansion=1, n_midlayers=1)       

        # for real-time inference
        self.kernels = None
        self.prev_out = 0.
        
    def forward(self, f0, loudness, periodicity, ema, spkr_emb):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model 
        loudness    : in shape [B, 1, t], calculated at f_model
        periodicity : in shape [B, 1, t], calculated at f_model
        ema         : in shape [B, 12, t], sampled at f_model    
        spkr_emb    : in shape [B, spkr_emb_dim]
        
        #### output ####
        speech      : in shape [B, t*framesize]
        
        """
        # going through encoder to get the control signals
        cn_sin, cn_cos, an_sin, an_cos, H = self.encoder(f0, loudness, periodicity, ema, spkr_emb)
        
        # generate harmonic components
        harmonics = self.harmonic(f0, cn_sin, cn_cos, an_sin, an_cos) # [B, t*framesize]
        
        # generate filtered noise
        noise = self.noise(H) # [B, t*framesize]
        
        # additive synthesis
        speech = harmonics + noise # [B, t*framesize]
        
        # reverb
        kernels = self.reverb_mlp(spkr_emb) # [B, reverb_len]
        speech = fftconvolve(speech, kernels, mode='full')[:, :-(self.reverb_len-1)] # [B, t*framesize]
        
        return speech
        
    @torch.no_grad()
    def realtime_forward(self, f0, f0_left_context, loudness, loudness_left_context, periodicity, periodicity_left_context, ema, ema_left_context, spkr_emb):
        """chunk size = t
        #### input ####
        f0                      : in shape [B, 1, t], calculated at f_model 
        f0_left_context         : in shape [B, 1, CONTEXT_LEN], calculated at f_model 
        loudness                : in shape [B, 1, t], calculated at f_model
        loudness_left_context   : in shape [B, 1, CONTEXT_LEN], calculated at f_model 
        periodicity             : in shape [B, 1, t], calculated at f_model
        periodicity_left_context: in shape [B, 1, CONTEXT_LEN], calculated at f_model 
        ema                     : in shape [B, 12, t], sampled at f_model   
        ema_left_context        : in shape [B, 12, CONTEXT_LEN], calculated at f_model  
        spkr_emb                : in shape [B, spkr_emb_dim]
        
        #### output ####
        out                     : in shape [B, t*framesize]
        
        """
        t = f0.shape[-1]
        
        # going through encoder to get the control signals
        f0_full = torch.concat([f0_left_context, f0], dim=-1)
        loudness_full = torch.concat([loudness_left_context, loudness], dim=-1)
        periodicity_full = torch.concat([periodicity_left_context, periodicity], dim=-1)
        ema_full = torch.concat([ema_left_context, ema], dim=-1)        
        
        cn_sin, cn_cos, an_sin, an_cos, H = self.encoder.realtime_forward(f0_full, loudness_full, periodicity_full, ema_full, spkr_emb)
        
        # generate harmonic components
        harmonics = self.harmonic.realtime_forward(f0, cn_sin, cn_cos, an_sin, an_cos) # [B, t*framesize]
        
        # generate filtered noise
        noise = self.noise.realtime_forward(H) # [B, t*framesize]
        
        # additive synthesis
        speech = harmonics + noise # [B, t*framesize]
    
        # reverb
        if self.kernels is None:
            self.kernels = self.reverb_mlp(spkr_emb) # [B, reverb_len]
        speech = fftconvolve(speech, self.kernels, mode='full') # [B, t*framesize+reverb_len-1]
        speech[:, :(self.reverb_len-1)] += self.prev_out
        
        out = speech[:, :t*self.framesize] # [B, t*framesize]
        self.prev_out = speech[:, -(self.reverb_len-1):] # [B, reverb_len-1]
        
        return out

    
    

        
        
