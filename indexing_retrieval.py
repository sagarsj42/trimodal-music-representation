#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('/scratch/sagarsj42')
os.environ['TRANSFORMERS_CACHE'] = '/scratch/sagarsj42'


# In[2]:


import faiss
import numpy as np
import pandas as pd


# In[3]:


DATASET_INFO_DIR = './yt8m-clips-dataset-info'
EMBEDS_DIR = 'weighted-contrastive-embeds'
EMB_SIZE = 300
RET_SIZE = 20000


# In[4]:


split = 'test'
media = ['text', 'audio', 'video']


# In[5]:


clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, split, 'clip-info.jsonl'), lines=True)

print(clip_df.info())

clip_df.head()


# In[6]:


m_embeds = dict()
m_indices = dict()
for m in media:
    embeds = list()
    for _, row in clip_df.iterrows():
        vid = row['vid']
        clip_no = row['clip_no']
        file_name = f'{vid}-{clip_no}-{m}-emb.npy'
        sample_embed = np.load(os.path.join(EMBEDS_DIR, split, m, file_name))
        embeds.append(sample_embed)
    m_embeds[m] = np.array(embeds)
    
    print(m, m_embeds[m].shape)
    
    index = faiss.IndexFlatIP(EMB_SIZE)
    index.add(m_embeds[m])
    m_indices[m] = index
    
    print('Index constructed')

m_indices


# In[7]:


for i in range(len(media)):
    for j in range(i+1, len(media)):
        m_1 = media[i]
        m_2 = media[j]
        print(f'Retrieval: {m_1} to {m_2}')
        _, res = m_indices[media[j]].search(m_embeds[media[i]], RET_SIZE)
        
        n_rows = res.shape[0]
        r_1 = 0
        r_5 = 0
        r_10 = 0
        ranks = list()
        for k in range(n_rows):
            search = res[k, :]
            try:
                pos = np.where(search == k)[0][0] + 1
            except IndexError:
                pos = RET_SIZE + 1
            if pos <= 1:
                r_1 += 1
            elif pos <= 5:
                r_5 =+ 1
            elif pos <= 10:
                r_10 += 1
            ranks.append(pos)
        ranks = np.array(ranks)
        mean_r = ranks.mean()
        median_r = np.median(ranks)
        r_1 = r_1 / n_rows * 100.0
        r_5 = r_5 / n_rows * 100.0
        r_10 = r_10 / n_rows * 100.0
        print(f'Recall @ 1: {r_1}, @ 5: {r_5}, @ 10: {r_10}')
        print(f'Mean rank: {mean_r}, median rank: {median_r}')
        print()


# In[ ]:




