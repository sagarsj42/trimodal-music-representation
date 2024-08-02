import torch
from torch import nn
from bpemb import BPEmb


class TextModel(nn.Module):
    def __init__(self, emb_size, bpe_vocab_size):
        super(TextModel, self).__init__()
        self.emb_size = emb_size
        self.bpe_vocab_size = bpe_vocab_size
        
        self.emb_model = BPEmb(lang='en', vs=self.bpe_vocab_size, dim=self.emb_size)
        self.model = nn.Sequential(
            nn.Embedding.from_pretrained(torch.tensor(self.emb_model.vectors)),
            nn.LayerNorm(self.emb_size),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.emb_size, self.emb_size)
        )
    
    def forward(self, x):
        mask = (x != 0) * 1
        n_tokens = torch.clamp(mask.sum(dim=1), min=1).unsqueeze(-1)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.emb_size)
        
        x = self.model(x)
        x = x * expanded_mask
        x = x.sum(dim=1)
        x = x / n_tokens
        
        return x


if __name__ == '__main__':
    from utils import get_n_params
    
    
    EMB_SIZE = 300
    BPE_VOCAB_SIZE = 10000
    
    text_model = TextModel(EMB_SIZE, BPE_VOCAB_SIZE)
    
    print(text_model)
    print('# text model params:', get_n_params(text_model))
