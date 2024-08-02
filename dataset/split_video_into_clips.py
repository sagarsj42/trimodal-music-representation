import os
import sys
import glob
import json

from moviepy.editor import VideoFileClip


data_home_dir = sys.argv[1]

DOWN_DATA_DIR = os.path.join(data_home_dir, 'yt8m-av-down-data')
CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-video-clips')
splits = ['dev', 'test']

threshold = 8
clip_off_last = 3
file_format = 'mp4'

for split in splits:
    input_split_dir = os.path.join(DOWN_DATA_DIR, split)
    output_split_dir = os.path.join(CLIPS_DATA_DIR, split)
    info_file_name = f'video-clips-info-{split}.jsonl'

    os.makedirs(output_split_dir, exist_ok=True)
    os.system(f"touch {CLIPS_DATA_DIR}/{info_file_name}")

    for vid in os.listdir(input_split_dir):
        if len(os.listdir(os.path.join(input_split_dir, vid))) != 2:
            continue
        video_file = glob.glob(pathname=f'{input_split_dir}/{vid}/{vid}-video.*')[0]
        try:
            print(vid)
            video_object = VideoFileClip(video_file)
        except OSError as e:
            print('Converting to mp4, video file:', video_file)
            vid_dir = os.path.join(input_split_dir, vid)
            filename = os.path.basename(video_file)
            new_filename = filename.split('.')[0] + '.mp4'
            os.system(f'cd {input_split_dir}/{vid} && MP4Box -add {filename} {new_filename} >/dev/null 2>&1')
            video_object = VideoFileClip(os.path.join(vid_dir, new_filename))

        dur = video_object.duration
        start = 0
        end = 0
        c = 0
        os.makedirs(os.path.join(output_split_dir, vid), exist_ok=True)

        while start < (dur - clip_off_last):
            end += threshold
            clip = video_object.subclip(start, min(end, dur))
            c += 1
            clip.write_videofile(os.path.join(output_split_dir, vid, f'{vid}-video-{c}.{file_format}'),
                                 verbose=False, logger=None)
            start += threshold
        info = {
            'vid': vid,
            'duration_sec': dur,
            'n_clips': c,
            'last_clip_dur': clip.duration
        }
        with open(os.path.join(CLIPS_DATA_DIR, info_file_name), 'a') as f:
            f.write(json.dumps(info) + '\n')
