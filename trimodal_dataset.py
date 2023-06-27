import os
import random
import string

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class CosineSimDataset(Dataset):
    def __init__(self, split, clips_info_path, text_bpe_model, audio_features_path, video_features_path):
        self.split = split
        self.clips_info_path = clips_info_path
        self.text_bpe_model = text_bpe_model
        self.audio_features_path = audio_features_path
        self.video_features_path = video_features_path
        
        self.vid_df = pd.read_json(os.path.join(self.clips_info_path, split, 
                            'video-info.jsonl'), lines=True)
        self.clips_df = pd.read_json(os.path.join(self.clips_info_path, self.split, 
                            'clip-info.jsonl'), lines=True)
        
        allowed_text = set(string.ascii_lowercase)
        all_texts = list()
        for _, row in self.vid_df.iterrows():
            text = [t.lower().strip() for t in row['labels'] + row['tags'].split(',')]
            text = filter(lambda t: len(t) > 2, text)
            text = list(filter(lambda t: set(t) <= set(allowed_text), text))
            all_texts.append(text)
        self.vid_df['texts'] = all_texts
        
        
    def __len__(self):
        return self.clips_df.shape[0]
    
    
    def __getitem__(self, idx):
        clips_row = self.clips_df.iloc[idx]
        vid = clips_row['vid']
        vid_row = self.vid_df[self.vid_df['vid'] == vid].iloc[0]
        split_dir = vid_row['split']
        
        try:
            text = random.choice(vid_row['texts'])
            text_ids = np.array(self.text_bpe_model.encode_ids(text))
        except IndexError:
            text_ids = np.array([0])
        audio_clip_filename = clips_row['audio_clip_name']
        audio_feat_filename = audio_clip_filename[:-4].replace('-audio-', '-audfeat-') + '.npy'
        video_clip_filename = clips_row['video_clip_name']
        video_feat_filename = video_clip_filename[:-4].replace('-video-', '-vidfeat-') + '.npy'
        audio_feat = np.load(os.path.join(self.audio_features_path, split_dir, vid, audio_feat_filename))
        video_feat = np.load(os.path.join(self.video_features_path, split_dir, vid, video_feat_filename))
    
        return {'text_inp': text_ids, 'audio_inp': audio_feat, 'video_inp': video_feat}

    
class CosineSimDatasetWithMD(Dataset):
    def __init__(self, split, clips_info_path, text_bpe_model, audio_features_path, video_features_path):
        self.split = split
        self.clips_info_path = clips_info_path
        self.text_bpe_model = text_bpe_model
        self.audio_features_path = audio_features_path
        self.video_features_path = video_features_path
        
        self.vid_df = pd.read_json(os.path.join(self.clips_info_path, split, 
                            'video-info.jsonl'), lines=True)
        self.clips_df = pd.read_json(os.path.join(self.clips_info_path, self.split, 
                            'clip-info.jsonl'), lines=True)
        
        allowed_text = set(string.ascii_lowercase)
        all_texts = list()
        for _, row in self.vid_df.iterrows():
            text = [t.lower().strip() for t in row['labels'] + row['tags'].split(',')]
            text = filter(lambda t: len(t) > 2, text)
            text = list(filter(lambda t: set(t) <= set(allowed_text), text))
            all_texts.append(text)
        self.vid_df['texts'] = all_texts
        
        
    def __len__(self):
        return self.clips_df.shape[0]
    
    
    def __getitem__(self, idx):
        clips_row = self.clips_df.iloc[idx]
        vid = clips_row['vid']
        clip_no = clips_row['clip_no']
        vid_row = self.vid_df[self.vid_df['vid'] == vid].iloc[0]
        split_dir = vid_row['split']
        
        try:
            text = random.choice(vid_row['texts'])
            text_ids = np.array(self.text_bpe_model.encode_ids(text))
        except IndexError:
            text_ids = np.array([0])
        audio_clip_filename = clips_row['audio_clip_name']
        audio_feat_filename = audio_clip_filename[:-4].replace('-audio-', '-audfeat-') + '.npy'
        video_clip_filename = clips_row['video_clip_name']
        video_feat_filename = video_clip_filename[:-4].replace('-video-', '-vidfeat-') + '.npy'
        audio_feat = np.load(os.path.join(self.audio_features_path, split_dir, vid, audio_feat_filename))
        video_feat = np.load(os.path.join(self.video_features_path, split_dir, vid, video_feat_filename))
    
        return {'text_inp': text_ids, 'audio_inp': audio_feat, 'video_inp': video_feat, 'vid': vid, 'clip_no': clip_no}


def collate_trimodal_cosine(batch):
    texts = list()
    audios = list()
    videos = list()
    max_t = max([len(s['text_inp']) for s in batch])
    audio_feats_len = 384000
    for sample in batch:
        text_ids = sample['text_inp']
        pad_len = max_t - len(text_ids)
        padded_text_ids = np.concatenate((text_ids, np.zeros(pad_len)))
        texts.append(torch.tensor(padded_text_ids, dtype=torch.long).unsqueeze(0))
        
        audio_feats = sample['audio_inp'][:audio_feats_len]
        pad_len = audio_feats_len - len(audio_feats)
        padded_audio_feats = np.concatenate((audio_feats, np.zeros(pad_len)))
        audios.append(torch.tensor(padded_audio_feats, dtype=torch.float32).unsqueeze(0))
        
        videos.append(torch.tensor(sample['video_inp'], dtype=torch.float32).unsqueeze(0))
    texts = torch.cat(texts, dim=0)
    audios = torch.cat(audios, dim=0)
    videos = torch.cat(videos, dim=0)
    
    return {'text_batch': texts, 'audio_batch': audios, 'video_batch': videos}


def collate_trimodal_with_metadata(batch):
    texts = list()
    audios = list()
    videos = list()
    vids = list()
    clip_nos = list()
    max_t = max([len(s['text_inp']) for s in batch])
    audio_feats_len = 384000
    for sample in batch:
        text_ids = sample['text_inp']
        pad_len = max_t - len(text_ids)
        padded_text_ids = np.concatenate((text_ids, np.zeros(pad_len)))
        texts.append(torch.tensor(padded_text_ids, dtype=torch.long).unsqueeze(0))
        
        audio_feats = sample['audio_inp'][:audio_feats_len]
        pad_len = audio_feats_len - len(audio_feats)
        padded_audio_feats = np.concatenate((audio_feats, np.zeros(pad_len)))
        audios.append(torch.tensor(padded_audio_feats, dtype=torch.float32).unsqueeze(0))
        
        videos.append(torch.tensor(sample['video_inp'], dtype=torch.float32).unsqueeze(0))
        vids.append(sample['vid'])
        clip_nos.append(sample['clip_no'])
    texts = torch.cat(texts, dim=0)
    audios = torch.cat(audios, dim=0)
    videos = torch.cat(videos, dim=0)
    
    return {'text_batch': texts, 'audio_batch': audios, 'video_batch': videos, 'vids': vids, 'clip_nos': clip_nos}


if __name__ == '__main__':
    os.chdir('/scratch/sagarsj42')
    
    from bpemb import BPEmb
    from torch.utils.data import DataLoader
    from icecream import ic
    
    
    DATASET_INFO_DIR = './yt8m-clips-dataset-info'
    AUDIO_FEATURES_DIR = './yt8m-audio-features'
    VIDEO_FEATURES_DIR = './yt8m-video-features'
    EMB_SIZE = 300
    BPE_VOCAB_SIZE = 10000
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 4
    
    text_bpe_model = BPEmb(lang='en', vs=BPE_VOCAB_SIZE, dim=EMB_SIZE)
    
    train_ds = CosineSimDataset('train', DATASET_INFO_DIR, text_bpe_model, AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    dev_ds = CosineSimDataset('dev', DATASET_INFO_DIR, text_bpe_model, AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    test_ds = CosineSimDataset('test', DATASET_INFO_DIR, text_bpe_model, AUDIO_FEATURES_DIR, VIDEO_FEATURES_DIR)
    
    ic(len(train_ds), len(dev_ds), len(test_ds))
    
    train_dl = DataLoader(train_ds, collate_fn=collate_trimodal_cosine, 
                          batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dev_dl = DataLoader(dev_ds, collate_fn=collate_trimodal_cosine, 
                          batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, collate_fn=collate_trimodal_cosine, 
                          batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
    ic(len(train_dl), len(dev_dl), len(test_dl))

    sample_batch = next(iter(train_dl))
    
    ic(sample_batch.keys())
    ic(sample_batch['text_batch'].shape, sample_batch['audio_batch'].shape, sample_batch['video_batch'].shape)
