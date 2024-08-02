import os
import sys
import json

import pandas as pd
from pydub import AudioSegment
from moviepy.editor import VideoFileClip


data_home_dir = sys.argv[1]

AUDIO_CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-audio-clips')
VIDEO_CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-video-clips')
DATASET_PATH = os.path.join(data_home_dir, 'yt8m-clips-dataset-info')
EXCLUDED_VIDS = os.path.join(data_home_dir, 'dataset-excluded-ids.txt')
splits = ['train', 'dev', 'test']

clip_infos = list()
vid_infos = list()
for split in splits:
    audio_info_filename = f'audio-clips-info-{split}.jsonl'
    audio_info_filepath = os.path.join(AUDIO_CLIPS_DATA_DIR, audio_info_filename)
    video_info_filename = f'video-clips-info-{split}.jsonl'
    video_info_filepath = os.path.join(VIDEO_CLIPS_DATA_DIR, video_info_filename)

    audio_info_df = pd.read_json(audio_info_filepath, lines=True)
    video_info_df = pd.read_json(video_info_filepath, lines=True)
    audio_vids = audio_info_df['vid'].tolist()
    video_vids = video_info_df['vid'].tolist()
    vids = list(set(audio_vids).intersection(set(video_vids)))

    for i, vid in enumerate(vids):
        if i > 0 and i % 100 == 0:
            print(split, i)
        audio_clips_dir = os.path.join(AUDIO_CLIPS_DATA_DIR, split, vid)
        video_clips_dir = os.path.join(VIDEO_CLIPS_DATA_DIR, split, vid)
        audio_info = audio_info_df[audio_info_df['vid'] == vid].iloc[0]
        video_info = video_info_df[video_info_df['vid'] == vid].iloc[0]
        audio_clip_filenames = os.listdir(audio_clips_dir)
        video_clip_filenames = os.listdir(video_clips_dir)
        clip_nos = [int(fn[:-4].split('-')[-1]) for fn in audio_clip_filenames]
        v_clip_nos = [int(fn[:-4].split('-')[-1]) for fn in video_clip_filenames]
        vid_clip_infos = list()

        try:
            assert audio_info['n_clips'] == video_info['n_clips']
            assert audio_info['n_sampled_clips'] == video_info['n_sampled_clips']
            assert audio_info['n_sampled_clips'] == len(audio_clip_filenames)
            assert video_info['n_sampled_clips'] == len(video_clip_filenames)
            assert set(clip_nos) == set(v_clip_nos)
        except AssertionError:
            with open(EXCLUDED_VIDS, 'a') as f:
                f.write(vid + f',{split}\n')
            continue

        audio_sampled_dur = 0
        video_sampled_dur = 0
        for c in clip_nos:
            audio_filename = f'{vid}-audio-{c}.mp3'
            audio_filepath = os.path.join(audio_clips_dir, audio_filename)
            audio_clip = AudioSegment.from_file(audio_filepath)
            audio_clip_dur = len(audio_clip) / 1000.0
            audio_sampled_dur += audio_clip_dur

            video_filename = f'{vid}-video-{c}.mp4'
            video_filepath = os.path.join(video_clips_dir, video_filename)
            video_clip = VideoFileClip(video_filepath)
            video_clip_dur = video_clip.duration
            video_sampled_dur += video_clip_dur

            try:
                assert abs(audio_clip_dur - video_clip_dur) < 1.0
            except AssertionError:
                with open(EXCLUDED_VIDS, 'a') as f:
                    f.write(vid + f',{split}\n')
                continue

            vid_clip_infos.append({
                'vid': vid,
                'clip_no': c,
                'audio_clip_name': audio_filename,
                'audio_clip_dur': audio_clip_dur,
                'video_clip_name': video_filename,
                'video_clip_dur': video_clip_dur
            })

        try:
            assert abs(audio_info['sampled_dur'] - audio_sampled_dur) < 4.0
            assert abs(video_info['sampled_dur'] - video_sampled_dur) < 4.0
        except AssertionError:
            with open(EXCLUDED_VIDS, 'a') as f:
                f.write(vid + f',{split}\n')
            continue

        clip_infos.extend(vid_clip_infos)
        info_row = info_df[info_df['vid'] == vid].iloc[0]
        vid_infos.append({
            'vid': vid,
            'n_clips': int(audio_info['n_clips']),
            'n_sampled_clips': int(audio_info['n_sampled_clips']),
            'audio_dur': float(audio_info['duration_sec']),
            'sampled_audio_dur': audio_sampled_dur,
            'video_dur': float(video_info['duration_sec']),
            'sampled_video_dur': video_sampled_dur,
            'split': split,
            'labels': labels_df[labels_df['vid'] == vid].iloc[0]['labels'],
            'title': info_row['title'],
            'description': info_row['description'],
            'tags': info_row['tags']
        })

    split_dataset_path = os.path.join(DATASET_PATH, split)
    os.makedirs(split_dataset_path, exist_ok=True)
    with open(os.path.join(split_dataset_path, 'clip-info.jsonl'), 'w') as f:
        f.write('\n'.join([json.dumps(c) for c in clip_infos]))
    with open(os.path.join(split_dataset_path, 'video-info.jsonl'), 'w') as f:
        f.write('\n'.join([json.dumps(v) for v in vid_infos]))
