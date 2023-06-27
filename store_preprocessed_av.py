import os
import time
import concurrent.futures

import librosa
import numpy as np
import pandas as pd
from transformers import VideoMAEImageProcessor, VideoMAEConfig

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Normalize
)

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)


DATASET_INFO_DIR = './yt8m-clips-dataset-info'
CLIP_INFO_FILENAME = 'clip-info.jsonl'
VID_INFO_FILENAME = 'video-info.jsonl'
AUDIO_CLIPS_DIR = './yt8m-audio-clips'
VIDEO_CLIPS_DIR = './yt8m-video-clips'
AUDIO_FEATURES_DIR = './yt8m-audio-features'
VIDEO_FEATURES_DIR = './yt8m-video-features'
VIDEO_MODEL_KEY = 'MCG-NJU/videomae-base'

clip_duration = 8.0
image_processor = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_KEY)
model_config = VideoMAEConfig.from_pretrained(VIDEO_MODEL_KEY)

image_mean = image_processor.image_mean
image_std = image_processor.image_std

if 'shortest_edge' in image_processor.size:
    height = width = image_processor.size['shortest_edge']
else:
    height = image_processor.size['height']
    width = image_processor.size['width']
resize_to = (height, width)

num_frames_to_sample = model_config.num_frames

video_transform = Compose(
    [
        ApplyTransformToKey(
            key='video',
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x/255.0),
                    Normalize(image_mean, image_std),
                    Resize(resize_to)
                ]
            )
        )
    ]
)


def save_video_features(split):
    video_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(VIDEO_CLIPS_DIR, split),
        clip_sampler=pytorchvideo.data.make_clip_sampler('uniform', clip_duration),
        decode_audio=False,
        transform=video_transform
    )
    
    os.makedirs(os.path.join(VIDEO_FEATURES_DIR, split), exist_ok=True)
    for i, video_sample in enumerate(iter(video_dataset)):
        # if i > 1000:
        #     break
        if i > 0 and i % 1000 == 0:
            print(split, i, 'videos')
        video_clip_filename = video_sample['video_name']
        vid = video_clip_filename[:11]
        video_frames = video_sample['video'].permute(1, 0, 2, 3).numpy().astype(np.float16)

        os.makedirs(os.path.join(VIDEO_FEATURES_DIR, split, vid), exist_ok=True)
        video_features_filename = video_clip_filename[:-4].replace('-video-', '-vidfeat-') + '.npy'
        np.save(os.path.join(VIDEO_FEATURES_DIR, split, vid, video_features_filename), video_frames)
    
    return


def save_audio_features(split):
    if 'train' in split:
        clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, 'train', CLIP_INFO_FILENAME), lines=True)
        vid_df = pd.read_json(os.path.join(DATASET_INFO_DIR, 'train', VID_INFO_FILENAME), lines=True)
    else:
        clip_df = pd.read_json(os.path.join(DATASET_INFO_DIR, split, CLIP_INFO_FILENAME), lines=True)
        vid_df = pd.read_json(os.path.join(DATASET_INFO_DIR, split, VID_INFO_FILENAME), lines=True)
    vids = vid_df[vid_df['split'] == split]['vid'].tolist()
    os.makedirs(os.path.join(AUDIO_FEATURES_DIR, split), exist_ok=True)
    
    print(split, len(vids))
    
    for i, vid in enumerate(vids):
        # if i > 50:
        #     break
        if i > 0 and i % 50 == 0:
            print(split, i, 'audios')
        audio_filenames = clip_df[clip_df['vid'] == vid]['audio_clip_name']
        for clip_file_name in audio_filenames:
            audio_clip_filepath = os.path.join(AUDIO_CLIPS_DIR, split, vid, clip_file_name)
            audio_data, _ = librosa.load(audio_clip_filepath, sr=48000)
            os.makedirs(os.path.join(AUDIO_FEATURES_DIR, split, vid), exist_ok=True)
            audio_features_filename = clip_file_name[:-4].replace('-audio-', '-audfeat-') + '.npy'
            np.save(os.path.join(AUDIO_FEATURES_DIR, split, vid, audio_features_filename), audio_data)
    
    return


if __name__ == '__main__':
    data_splits = ['dev', 'test', 'train1', 'train2', 'train3', 'train4', 'train5']

    # print('Preprocessing video clips & saving features')

    # start = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
    #     futures = (pool.submit(save_video_features, current_split) 
    #             for current_split in data_splits)
    #     concurrent.futures.wait(futures)

    # print('Video preprocessing complete')
    # print('Time taken:', time.time() - start, 's')
    print('Preprocessing audio clips & saving features')

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = (pool.submit(save_audio_features, current_split) 
                for current_split in data_splits)
        concurrent.futures.wait(futures)

    print('Audio preprocessing complete')
    print('Time taken:', time.time() - start, 's')
