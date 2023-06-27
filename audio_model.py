import os
import sys

import torch
from torch import nn

import laion_clap
import transformers


class AudioModel(nn.Module):
    def __init__(self, emb_size, audio_model_key, audio_model_path='', quantize_input=True):
        super(AudioModel, self).__init__()
        self.emb_size = emb_size
        self.audio_model_key = audio_model_key
        self.audio_model_path = audio_model_path
        self.quantize = quantize_input
        
        transformers.logging.set_verbosity_error()
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.encoder = laion_clap.CLAP_Module(amodel=self.audio_model_key, enable_fusion=False)
        if self.audio_model_path:
            self.encoder.load_ckpt(self.audio_model_path)
        del self.encoder.model.text_branch
        del self.encoder.model.text_transform
        del self.encoder.model.text_projection
        sys.stdout.close()
        sys.stdout = self._original_stdout
        transformers.logging.set_verbosity_warning()
        
        self.encoder_out_size = self.encoder.model.audio_projection[2].out_features
        self.projector = nn.Sequential(
            nn.LayerNorm(self.encoder_out_size),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encoder_out_size, self.emb_size)
        )
    
    def forward(self, x):
        if self.quantize:
            x = self.int16_to_float32(self.float32_to_int16(x))
        x = self.encoder.get_audio_embedding_from_data(x, use_tensor=True)
        x = self.projector(x)
        
        return x
    
    def int16_to_float32(self, x):
        return (x / 32767.0).type(torch.float32)

    def float32_to_int16(self, x):
        x = torch.clamp(x, min=-1., max=1.)
        return (x * 32767.).type(torch.int16)


if __name__ == '__main__':
    from utils import get_n_params
    
    
    EMB_SIZE = 300
    AUDIO_MODEL_KEY = 'HTSAT-base'
    AUDIO_MODEL_PATH = '/scratch/sagarsj42/music_audioset_epoch_15_esc_90.14.pt'
    if not os.path.exists(AUDIO_MODEL_PATH):
        os.system('wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt')
    
    audio_model = AudioModel(EMB_SIZE, AUDIO_MODEL_KEY, AUDIO_MODEL_PATH).to('cpu')
    
    print(audio_model)
    print('# audio model params:', get_n_params(audio_model))
