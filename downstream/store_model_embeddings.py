#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('/scratch/sagarsj42')
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42'


# In[2]:


import random

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from bpemb import BPEmb

from tri_model import TriModel
from trimodal_dataset import CosineSimDatasetWithMD, collate_trimodal_with_metadata


# In[3]:


SEED = 15
EXP_NAME = 'weighted-contrastive'
DATASET_INFO_DIR = './yt8m-clips-dataset-info'
AUDIO_FEATURES_DIR = './yt8m-audio-features'
VIDEO_FEATURES_DIR = './yt8m-video-features'
EMB_SIZE = 300
BPE_VOCAB_SIZE = 10000
BATCH_SIZE = 8

EMBEDS_DIR = f'{EXP_NAME}-embeds'
# EMBEDS_DIR = 'zeroshot-embeds'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

split = 'test'

EMBEDS_DIR, DEVICE


# In[4]:


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[5]:


text_bpe_model = BPEmb(lang='en', vs=BPE_VOCAB_SIZE, dim=EMB_SIZE)
ds = CosineSimDatasetWithMD(split, DATASET_INFO_DIR, text_bpe_model, 
            AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)

len(ds)


# In[6]:


dl = DataLoader(ds, collate_fn=collate_trimodal_with_metadata, batch_size=BATCH_SIZE, shuffle=False)

len(dl)


# In[7]:


sample_batch = next(iter(dl))

sample_batch.keys()


# In[8]:


sample_batch['text_batch'].shape, sample_batch['audio_batch'].shape, sample_batch['video_batch'].shape, \
sample_batch['vids'], sample_batch['clip_nos']


# In[9]:


ckpt = torch.load(os.path.join(EXP_NAME, 'best.pth'))
model_args = ckpt['model_args']

model_args


# In[10]:


model = TriModel(**model_args)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)

model


# In[11]:


inp_batch = {
    'text_batch': sample_batch['text_batch'].to(DEVICE),
    'audio_batch': sample_batch['audio_batch'].to(DEVICE),
    'video_batch': sample_batch['video_batch'].to(DEVICE)
}
with torch.no_grad():
    embeds = model(*inp_batch.values())

embeds


# In[ ]:


for split in ['test', 'dev', 'train']:
    ds = CosineSimDatasetWithMD(split, DATASET_INFO_DIR, text_bpe_model, 
                AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    dl = DataLoader(ds, collate_fn=collate_trimodal_with_metadata, batch_size=BATCH_SIZE, shuffle=False)

    os.makedirs(os.path.join(EMBEDS_DIR, split, 'text'), exist_ok=True)
    os.makedirs(os.path.join(EMBEDS_DIR, split, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(EMBEDS_DIR, split, 'video'), exist_ok=True)

    for batch in tqdm(dl):
        vids = batch['vids']
        clip_nos = batch['clip_nos']
        inp_batch = {
            'text_batch': batch['text_batch'].to(DEVICE),
            'audio_batch': batch['audio_batch'].to(DEVICE),
            'video_batch': batch['video_batch'].to(DEVICE)
        }

        with torch.no_grad():
            embeds = model(*inp_batch.values())

        for i in range(len(vids)):
            vid = vids[i]
            clip_no = clip_nos[i]
            te = embeds['text_emb'][i, :].cpu().numpy()
            ae = embeds['audio_emb'][i, :].cpu().numpy()
            ve = embeds['video_emb'][i, :].cpu().numpy()

            np.save(os.path.join(EMBEDS_DIR, split, 'text', f'{vid}-{clip_no}-text-emb.npy'), te)
            np.save(os.path.join(EMBEDS_DIR, split, 'audio', f'{vid}-{clip_no}-audio-emb.npy'), ae)
            np.save(os.path.join(EMBEDS_DIR, split, 'video', f'{vid}-{clip_no}-video-emb.npy'), ve)


# In[ ]:




