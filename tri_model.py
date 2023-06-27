from torch import nn

from text_model import TextModel
from audio_model import AudioModel
from video_model import VideoModel


class TriModel(nn.Module):
    def __init__(self, emb_size, bpe_vocab_size, audio_model_key, 
                 audio_model_path, video_model_key):
        
        super(TriModel, self).__init__()
        self.emb_size = emb_size
        self.bpe_vocab_size = bpe_vocab_size
        self.audio_model_key = audio_model_key
        self.audio_model_path = audio_model_path
        self.video_model_key = video_model_key
        
        self.text_model = TextModel(self.emb_size, self.bpe_vocab_size)
        self.audio_model = AudioModel(self.emb_size, self.audio_model_key, self.audio_model_path, 
                              quantize_input=True).to('cpu')
        self.video_model = VideoModel(self.emb_size, self.video_model_key)
    
    def forward(self, t_x, a_x, v_x):
        t_x = self.text_model(t_x)
        a_x = self.audio_model(a_x)
        v_x = self.video_model(v_x)
        
        return {'text_emb': t_x, 'audio_emb': a_x, 'video_emb': v_x}


if __name__ == '__main__':
    import os
    from utils import get_n_params
    
    
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42'
    
    EMB_SIZE = 300
    BPE_VOCAB_SIZE = 10000
    AUDIO_MODEL_KEY = 'HTSAT-base'
    AUDIO_MODEL_PATH = '/scratch/sagarsj42/music_audioset_epoch_15_esc_90.14.pt'
    VIDEO_MODEL_KEY = 'MCG-NJU/videomae-base'
    
    if not os.path.exists(AUDIO_MODEL_PATH):
        os.system('wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt')
    
    trimodel = TriModel(EMB_SIZE, BPE_VOCAB_SIZE, AUDIO_MODEL_KEY, AUDIO_MODEL_PATH, VIDEO_MODEL_KEY).to('cpu')
    
    print(trimodel)
    print('# text model params:', get_n_params(trimodel.text_model))
    print('# audio model params:', get_n_params(trimodel.audio_model))
    print('# video model params:', get_n_params(trimodel.video_model))
    print('# trimodal model params:', get_n_params(trimodel))

