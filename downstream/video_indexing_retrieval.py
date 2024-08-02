#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('../../../data')


# In[3]:


import faiss
import numpy as np
import pandas as pd


# In[27]:


DATASET_INFO_DIR = './yt8m-clips-dataset-info'
EMBEDS_DIR = 'weighted-contrastive-embeds'
EMB_SIZE = 300
RET_SIZE = 10000


# In[28]:


splits = ['train', 'dev', 'test']
media = ['text', 'audio', 'video']


# In[6]:


clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, '../train', 'clip-info.jsonl'), lines=True)

print(clip_df.info())

clip_df.head()


# In[7]:


vid_df = pd.read_json(os.path.join(DATASET_INFO_DIR, '../train', 'video-info.jsonl'), lines=True)

print(vid_df.info())

vid_df.head()


# In[16]:


video_features_data = dict()
for split in splits:
    split_data = list()
    vid_df = pd.read_json(os.path.join(DATASET_INFO_DIR, split, 'video-info.jsonl'), lines=True)
    clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, split, 'clip-info.jsonl'), lines=True)
    for _, row in vid_df.iterrows():
        vid = row['vid']
        labels = row['labels']
        
        if split == 'train':
            n_clips = row['n_sampled_clips']
        else:
            n_clips = row['n_clips']
        clip_nos = clip_df[clip_df['vid'] == vid]['clip_no'].tolist()
        try:
            assert n_clips == len(clip_nos)
        except:
            print(f'Insufficient clips in {split} for {vid}: expected {n_clips}, found {len(clip_nos)}')
        
        text_embeds = list()
        audio_embeds = list()
        video_embeds = list()
        for clip_no in clip_nos:
            text_file_name = f'{vid}-{clip_no}-text-emb.npy'
            text_embed = np.load(os.path.join(EMBEDS_DIR, split, 'text', text_file_name))
            text_embeds.append(text_embed)
            audio_file_name = f'{vid}-{clip_no}-audio-emb.npy'
            audio_embed = np.load(os.path.join(EMBEDS_DIR, split, 'audio', audio_file_name))
            audio_embeds.append(audio_embed)
            video_file_name = f'{vid}-{clip_no}-video-emb.npy'
            video_embed = np.load(os.path.join(EMBEDS_DIR, split, 'video', video_file_name))
            video_embeds.append(video_embed)
        text_embeds = np.array(text_embeds)
        audio_embeds = np.array(audio_embeds)
        video_embeds = np.array(video_embeds)
        split_data.append({
            'vid': vid,
            'text_features': text_embeds,
            'audio_features': audio_embeds,
            'video_features': video_embeds,
            'labels': labels
        })
    video_features_data[split] = split_data

video_features_data.keys()


# In[17]:


len(video_features_data['train']), len(video_features_data['dev']), len(video_features_data['test'])


# In[18]:


video_features_data['train'][0]


# In[19]:


video_features_data['train'][0]['video_features'].mean(axis=0).shape


# In[20]:


mean_data = dict()
for split in splits:
    split_data = video_features_data[split]
    all_text = list()
    all_audio = list()
    all_video = list()
    all_labels = list()
    for instance in split_data:
        all_text.append(instance['text_features'].mean(axis=0))
        all_audio.append(instance['audio_features'].mean(axis=0))
        all_video.append(instance['video_features'].mean(axis=0))
        all_labels.append(instance['labels'])
    all_text = np.array(all_text)
    all_audio = np.array(all_audio)
    all_video = np.array(all_video)
    mean_data[split] = {'text': all_text, 'audio': all_audio, 'video': all_video, 'labels': all_labels}

mean_data.keys()


# In[21]:


split = 'test'
mean_data[split].keys()


# In[22]:


mean_data[split]['text'].shape, mean_data[split]['audio'].shape, mean_data[split]['video'].shape


# In[25]:


m_indices = dict()
for m in media:
    index = faiss.IndexFlatIP(EMB_SIZE)
    index.add(mean_data[split][m])
    m_indices[m] = index

m_indices


# In[31]:


for i in range(len(media)):
    for j in range(i+1, len(media)):
        m_1 = media[i]
        m_2 = media[j]
        print(f'Retrieval: {m_1} to {m_2}')
        _, res = m_indices[media[j]].search(mean_data[split][media[i]], RET_SIZE)
        
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
            if pos <= 5:
                r_5 += 1
            if pos <= 10:
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




