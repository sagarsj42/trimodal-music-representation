import os
import sys
import glob
import math
import json
import random

import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip


data_home_dir = sys.argv[1]

DOWN_DATA_DIR = os.path.join(data_home_dir, 'yt8m-av-down-data')
AUDIO_CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-audio-clips')
VIDEO_CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-video-clips')
split = 'train'

SEED_VALUE = 15
threshold = 8
clip_off_first = 4
clip_off_last = 4
audio_format = 'mp3'
video_format = 'mp4'

random.seed(SEED_VALUE)

input_split_dir = os.path.join(DOWN_DATA_DIR, split)
audio_output_split_dir = os.path.join(AUDIO_CLIPS_DATA_DIR, split)
video_output_split_dir = os.path.join(VIDEO_CLIPS_DATA_DIR, split)
audio_info_file_name = f'audio-clips-info-{split}.jsonl'
video_info_file_name = f'video-clips-info-{split}.jsonl'
skipped_vids_file_name = f'skipped-vids-{split}.txt'

os.makedirs(audio_output_split_dir, exist_ok=True)
os.makedirs(video_output_split_dir, exist_ok=True)
os.system(f"touch ./{AUDIO_CLIPS_DATA_DIR}/{audio_info_file_name}")
os.system(f"touch ./{VIDEO_CLIPS_DATA_DIR}/{video_info_file_name}")
os.system(f"touch ./{skipped_vids_file_name}")

done_a = list()
with open(os.path.join(AUDIO_CLIPS_DATA_DIR, audio_info_file_name), 'r') as f:
    for l in f.read().split('\n'):
        try:
            done_a.append(json.loads(l)['vid'])
        except:
            print(l)

done_v = list()
with open(os.path.join(VIDEO_CLIPS_DATA_DIR, video_info_file_name), 'r') as f:
    for l in f.read().split('\n'):
        try:
            done_v.append(json.loads(l)['vid'])
        except:
            print(l)

for vid in os.listdir(input_split_dir):
    if vid in done_a:
        continue
    print(vid, end='\t')
    if len(os.listdir(os.path.join(input_split_dir, vid))) < 2:
        with open(skipped_vids_file_name, 'a') as f:
            f.write(vid + f' | # files, {len(os.listdir(os.path.join(input_split_dir, vid)))} < 2' + '\n')
        continue

    audio_file = glob.glob(pathname=f'{input_split_dir}/{vid}/{vid}-audio.*')[0]
    audio_segment = AudioSegment.from_file(audio_file)

    a_dur_ms = len(audio_segment)
    a_dur_sec = len(audio_segment) / 1000
    a_start = clip_off_first * 1000
    a_end = a_start
    a_clips = list()
    while a_start < (a_dur_ms - clip_off_last * 1000):
        a_end += threshold * 1000
        clip = audio_segment[a_start:a_end]
        a_clips.append(clip)
        a_start += threshold * 1000

    video_file = glob.glob(pathname=f'{input_split_dir}/{vid}/{vid}-video.*')[0]
    try:
        video_object = VideoFileClip(video_file)
    except OSError as e:
        print('Converting to mp4, video file:', video_file)
        vid_dir = os.path.join(input_split_dir, vid)
        filename = os.path.basename(video_file)
        new_filename = filename.split('.')[0] + '.mp4'
        os.system(f'cd {input_split_dir}/{vid} && MP4Box -add {filename} {new_filename} >/dev/null 2>&1')
        video_object = VideoFileClip(os.path.join(vid_dir, new_filename))

    v_dur = video_object.duration
    v_start = clip_off_first
    v_end = v_start
    v_clips = list()
    while v_start < (v_dur - clip_off_last):
        v_end += threshold
        clip = video_object.subclip(v_start, min(v_end, v_dur))
        v_clips.append(clip)
        v_start += threshold

    if len(a_clips) != len(v_clips):
        with open(skipped_vids_file_name, 'a') as f:
            f.write(vid + f' | # audio clips, {len(a_clips)} != # video clips, {len(v_clips)}' + '\n')
        continue

    n_sparsity_options = [math.ceil(v_dur / (8*5)), math.ceil(v_dur / (8*4)), math.ceil(v_dur / (8*3)),
                          math.ceil(v_dur / (8*2)), math.ceil(v_dur / (8*1.5)), math.ceil(v_dur / 8)]
    n_sparsity_probs = [0.3, 0.2, 0.2, 0.2, 0.05, 0.05]
    n_sample_clips = np.random.choice(n_sparsity_options, p=n_sparsity_probs)
    clip_indxs = random.sample(range(len(a_clips)), n_sample_clips)
    print(len(a_clips), len(v_clips), n_sample_clips)

    sampled_a_clips = [a_clips[i] for i in clip_indxs]
    sampled_v_clips = [v_clips[i] for i in clip_indxs]
    a_sampled_dur_sec = sum([len(c) for c in sampled_a_clips]) / 1000.0
    v_sampled_dur_sec = sum([c.duration for c in sampled_v_clips])

    os.makedirs(os.path.join(audio_output_split_dir, vid), exist_ok=True)
    os.makedirs(os.path.join(video_output_split_dir, vid), exist_ok=True)
    for ci, ac, vc in zip(clip_indxs, a_clips, v_clips):
        ac.export(os.path.join(audio_output_split_dir, vid, f'{vid}-audio-{ci + 1}.{audio_format}'),
                  format=audio_format)
        vc.write_videofile(os.path.join(video_output_split_dir, vid, f'{vid}-video-{ci + 1}.{video_format}'),
                           verbose=False, logger=None)

    a_info = {
        'vid': vid,
        'duration_sec': a_dur_sec,
        'n_clips': len(a_clips),
        'n_sampled_clips': n_sample_clips,
        'sampled_dur': a_sampled_dur_sec
    }
    v_info = {
        'vid': vid,
        'duration_sec': v_dur,
        'n_clips': len(v_clips),
        'n_sampled_clips': n_sample_clips,
        'sampled_dur': v_sampled_dur_sec
    }
    with open(os.path.join(AUDIO_CLIPS_DATA_DIR, audio_info_file_name), 'a') as f:
        f.write(json.dumps(a_info) + '\n')
    with open(os.path.join(VIDEO_CLIPS_DATA_DIR, video_info_file_name), 'a') as f:
        f.write(json.dumps(v_info) + '\n')
