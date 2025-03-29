import torch
import torch.nn as nn
from torchaudio.functional import fftconvolve

class FilteredNoiseGenerator(nn.Module):
    def __init__(self, framesize=80, nbands=257):
        super().__init__()
        self.framesize = framesize
        self.prev_out = 0.
        self.register_buffer("hann", torch.hann_window(2*nbands-1).float())
        
    def forward(self, H):
        """Generate time-varying filtered noise by overlap-add method using linear phase LTV-FIR filter
        #### input ####
        H   : half of the frequency response of the zero-phase filter, all real numbers
                in shape [B, t, nbands], sampled at f_model = fs / framesize, default to fs = 16kHz, f_model = 200Hz
        
        #### output ####
        out : in shape [B, t*framesize]
    
        """
        B, t, nbands = H.shape
        
        # irfft to get zero-phase filter
        zero_phase = torch.fft.irfft(H, n=2*nbands-1, dim=-1) # [B, t, 2*nbands-1]
        
        # shift zero-phase filter to causal
        lin_phase = zero_phase.roll(nbands-1, -1) 
        
        # window the filter in the time domain
        firwin = lin_phase * self.hann # [B, t, 2*nbands-1]
        L = firwin.shape[-1]
            
        # generate noise
        noise = torch.rand(B, t, self.framesize).float().to(H.device) * 2 - 1 # [B, t, framesize]
        
        # fftconvolve
        filtered_noise = fftconvolve(noise, firwin) # [B, t, framesize+L-1]
        
        # Overlap-add to build time-varying filtered noise.
        eye = torch.eye(filtered_noise.shape[-1]).unsqueeze(1).to(H.device) # [framesize+L-1, 1, framesize+L-1]
        out = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), eye, stride=self.framesize).squeeze(1) # [B, t*framesize+L-1]
        out = out[:, :-(L-1)] # [B, t*framesize]
        return out
    
    def realtime_forward(self, H):
        # chunk size = t
        B, t, nbands = H.shape
        zero_phase = torch.fft.irfft(H, n=2*nbands-1, dim=-1) # [B, t, 2*nbands-1]
        
        # shift zero-phase filter to causal
        lin_phase = zero_phase.roll(nbands-1, -1)
        
        # window the filter in the time domain
        firwin = lin_phase * self.hann # [B, t, 2*nbands-1]
        L = firwin.shape[-1] # 2*nbands-1
        
        # generate noise
        noise = torch.rand(B, t, self.framesize).float().to(H.device) * 2 - 1 # [B, t, framesize]
        
        # generate new output with overlap and add
        filtered_noise = fftconvolve(noise, firwin) # [B, t, framesize+L-1]
        eye = torch.eye(filtered_noise.shape[-1]).unsqueeze(1).to(H.device) # [framesize+L-1, 1, framesize+L-1]
        out = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), eye, stride=self.framesize).squeeze(1) # [B, t*framesize+L-1]
        out[:, :(L-1)] += self.prev_out
        
        # update prev_out
        self.prev_out = out[:, -(L-1):] # [B, L-1]
        
        return out[:, :t*self.framesize] 
    
    def clear(self):
        self.prev_out = 0
    