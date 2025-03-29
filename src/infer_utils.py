import torch
from torchaudio.transforms import Spectrogram
import soundfile as sf
import librosa
from IPython.display import Audio

class VC(object):
    def __init__(self, framesize, vocoder, spkr_encoder, pitch_tracker, ema_inverter, device):
        # default all models are in eval mode!
        
        self.framesize = framesize
        self.device = device
        self.vocoder = vocoder.to(device)
        self.spkr_encoder = spkr_encoder.to(device)
        self.pitch_tracker = pitch_tracker.to(device)
        self.ema_inverter = ema_inverter.to(device)
    
    def load_wav(self, wav, fs=None):
        if isinstance(wav, str):
            wav, fs = sf.read(wav)
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if fs != 16000 and fs != None:
            wav = librosa.resample(wav, orig_sr=fs, target_sr=16000)
        return torch.from_numpy(wav)
    
    def get_loudness_spec(self, wav, power=1, n_fft=1024, fs=None):
        """
        #### input ####
        wav         : a path (str) or a tensor / np.array in shape [t, ]
        
        #### output ####
        loudness    : a tensor in shape [t_model, ]
        
        """
        wav = self.load_wav(wav, fs)
        wav = wav.to(self.device).float()
        spec = Spectrogram(n_fft=n_fft, hop_length=self.framesize, power=power).to(self.device)
        loudness = spec(wav).mean(0)
        return loudness
    
    def get_pitch(self, wavpath):
        """
        #### input ####
        wavpath                  : a path (str)
        
        #### output ####
        out['pitch']             : a tensor in shape [1, 1, t_model]
        out['periodicity']       : a tensor in shape [1, 1, t_model]
        out['loudness']          : a tensor in shape [1, 1, t_model]
        
        """
        out = self.pitch_tracker.infer_from_file(wavpath, self.device)
        return out['pitch'], out['periodicity'], out['loudness']
    
    def get_ema(self, wavpath):
        """
        #### input ####
        wavpath : a path (str) 
        
        #### output ####
        ema     : a tensor in shape [1, 12, t_model]
        
        """
        ema = self.ema_inverter.infer_from_file(wavpath, self.device)
        return ema
    
    @torch.no_grad()
    def get_spkr_emb(self, wavpath, periodicity=None, fs=None):
        """
        #### input ####
        wavpath     : a path (str)
        periodicity : in shape [1, 1, t_model], sampled @200Hz
        
        #### output ####
        spkr_emb    : in shape [1, spkr_emb_dim]
        
        """
        wav = self.load_wav(wavpath, fs)
        wav = wav.to(self.device).float().unsqueeze(0) # [1, t]

        if periodicity is None:
            _, periodicity, _ = self.get_pitch_full(wavpath)
        spkr_emb = self.spkr_encoder(wav, periodicity)
            
        return spkr_emb        
    
    def get_f0_median(self, pitch, periodicity):
        """
        #### input ####
        pitch             : a tensor in shape [1, 1, t_model]
        periodicity       : a tensor in shape [1, 1, t_model]
        
        #### output ####
        pitch_median      : a tensor in shape [1, ]
        
        """
        # extract voiced f0
        pitch = pitch[periodicity > 0.5].squeeze()
        return torch.median(pitch)
    
    @torch.no_grad()
    def get_tgt_info(self, tgt_path):
        wav = self.load_wav(tgt_path)
        wav = wav.to(self.device).float().unsqueeze(0) # [1, t]
        
        # extract target spkr f0 median
        pitch, periodicity = self.get_pitch(tgt_path)
        pitch = pitch[periodicity > 0.5].squeeze()
        tgt_f0_median = torch.median(pitch) # [1, ]
        
        # extract target spkr emb
        spkr_emb = self.spkr_encoder(wav, periodicity) # [1, spkr_emb_dim]
        
        return spkr_emb, tgt_f0_median

    @torch.no_grad()
    def vc(self, src_path, tgt_path):
        # ema
        ema = self.get_ema(src_path) # [1, 12, t]
        fix_len = ema.shape[-1]
        
        # pitch, periodicity and loudness
        src_pitch, src_periodicity, src_loudness = self.get_pitch(src_path) # [1, 1, t]
        tgt_pitch, tgt_periodicity, tgt_loudness = self.get_pitch(tgt_path) # [1, 1, t]
        src_pitch = src_pitch[:, :, :fix_len] # [1, 1, t]
        src_periodicity = src_periodicity[:, :, :fix_len] # [1, 1, t]
        loudness = src_loudness[:, :, :fix_len] # [1, 1, t]
        
        # pitch median
        median_src = self.get_f0_median(src_pitch, src_periodicity)
        median_tgt = self.get_f0_median(tgt_pitch, tgt_periodicity)
        f0 = src_pitch * (median_tgt / median_src) # [1, 1, t]
        
        # spkr emb
        spkr_emb_tgt = self.get_spkr_emb(tgt_path, periodicity=tgt_periodicity) # [1, spkr_emb_dim]
        
        # feed into the model
        transformed = self.vocoder(f0, loudness, src_periodicity, ema, spkr_emb_tgt)
        
        return transformed.squeeze().cpu().numpy()

def play_audio(wav, fs=None):
    if isinstance(wav, str):
        wav, sr = sf.read(wav)
        if fs is not None and sr != fs:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=fs)
        else:
            fs = sr
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().cpu()
    return Audio(wav, rate=fs)
