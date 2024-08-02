import transformers
from transformers import VideoMAEModel

from torch import nn


class VideoModel(nn.Module):
    def __init__(self, emb_size, video_model_key):
        super(VideoModel, self).__init__()
        self.emb_size = emb_size
        self.video_model_key = video_model_key
        
        transformers.logging.set_verbosity_error()
        self.encoder = VideoMAEModel.from_pretrained(self.video_model_key)
        self.encoder_out_size = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.LayerNorm(self.encoder_out_size),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encoder_out_size, self.emb_size)
        )
        transformers.logging.set_verbosity_warning()
    
    def forward(self, x):
        x = self.encoder(pixel_values=x).last_hidden_state
        x = x.mean(dim=1)
        x = self.projector(x)
        
        return x


if __name__ == '__main__':
    import os
    from utils import get_n_params
    
    
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42'
    EMB_SIZE = 300
    VIDEO_MODEL_KEY = 'MCG-NJU/videomae-base'
    
    video_model = VideoModel(EMB_SIZE, VIDEO_MODEL_KEY)
    
    print(video_model)
    print('# video model params:', get_n_params(video_model))
