#!/usr/bin/env python
# coding: utf-8

# In[8]:


# In[9]:


import os
os.chdir('../../data')


# In[37]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


# In[12]:


DATASET_INFO_DIR = './yt8m-clips-dataset-info'
EMBEDS_DIR = 'weighted-contrastive-embeds'


# In[13]:


splits = ['train', 'dev', 'test']
media = ['text', 'audio', 'video']


# In[14]:


clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, 'train', 'clip-info.jsonl'), lines=True)

print(clip_df.info())

clip_df.head()


# In[15]:


vid_df = pd.read_json(os.path.join(DATASET_INFO_DIR, 'train', 'video-info.jsonl'), lines=True)

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
        
        audio_embeds = list()
        video_embeds = list()
        for clip_no in clip_nos:
            audio_file_name = f'{vid}-{clip_no}-audio-emb.npy'
            audio_embed = np.load(os.path.join(EMBEDS_DIR, split, 'audio', audio_file_name))
            audio_embeds.append(audio_embed)
            video_file_name = f'{vid}-{clip_no}-video-emb.npy'
            video_embed = np.load(os.path.join(EMBEDS_DIR, split, 'video', video_file_name))
            video_embeds.append(video_embed)
        audio_embeds = np.array(audio_embeds)
        video_embeds = np.array(video_embeds)
        split_data.append({
            'vid': vid,
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
    all_audio = list()
    all_video = list()
    all_labels = list()
    for instance in split_data:
        all_audio.append(instance['audio_features'].mean(axis=0))
        all_video.append(instance['video_features'].mean(axis=0))
        all_labels.append(instance['labels'])
    all_audio = np.array(all_audio)
    all_video = np.array(all_video)
    mean_data[split] = {'audio': all_audio, 'video': all_video, 'labels': all_labels}

mean_data.keys()


# In[21]:


mean_data['train'].keys()


# In[22]:


mean_data['train']['audio'].shape, mean_data['train']['video'].shape, len(mean_data['train']['labels'])


# In[23]:


all_labels = set()
[all_labels.update(labels) for labels in mean_data['train']['labels']]
len(all_labels)
label_enc = LabelEncoder()
label_enc.fit(list(all_labels))


# In[24]:


multi_lab_bin = MultiLabelBinarizer()
mean_data['train']['mult_lab'] = multi_lab_bin.fit_transform(
    [label_enc.transform(labels) for labels in mean_data['train']['labels']])

mean_data['train']['mult_lab'].shape


# In[25]:


mean_data['dev']['mult_lab'] = multi_lab_bin.transform(
    [label_enc.transform(labels) for labels in mean_data['dev']['labels']])

mean_data['dev']['mult_lab'].shape


# In[26]:


mean_data['test']['mult_lab'] = multi_lab_bin.transform(
    [label_enc.transform(labels) for labels in mean_data['test']['labels']])

mean_data['test']['mult_lab'].shape


# In[27]:


x_train = np.hstack((mean_data['train']['audio'], mean_data['train']['video']))
x_dev = np.hstack((mean_data['dev']['audio'], mean_data['dev']['video']))
x_test = np.hstack((mean_data['test']['audio'], mean_data['test']['video']))

y_train = mean_data['train']['mult_lab']
y_dev = mean_data['dev']['mult_lab']
y_test = mean_data['test']['mult_lab']

x_train.shape, x_dev.shape, x_test.shape, y_train.shape, y_dev.shape, y_test.shape


# In[33]:


mlp = MLPClassifier(hidden_layer_sizes=(300,100), max_iter=1000)
mlp.fit(x_train, y_train)


# In[36]:


pred_train = mlp.predict(x_train)
pred_dev = mlp.predict(x_dev)
pred_test = mlp.predict(x_test)

pred_train.shape, pred_dev.shape, pred_test.shape


# In[38]:


print('train accuracy:', accuracy_score(y_true=y_train, y_pred=pred_train))


# In[39]:


print('dev accuracy:', accuracy_score(y_true=y_dev, y_pred=pred_dev))


# In[40]:


print('test accuracy:', accuracy_score(y_true=y_test, y_pred=pred_test))
print('test f1:', f1_score(y_true=y_test, y_pred=pred_test, average='weighted'))
print('test precision:', precision_score(y_true=y_test, y_pred=pred_test, average='weighted'))
print('test recall:', recall_score(y_true=y_test, y_pred=pred_test, average='weighted'))
print('test hamming loss:', hamming_loss(y_true=y_test, y_pred=pred_test))


# In[ ]:




