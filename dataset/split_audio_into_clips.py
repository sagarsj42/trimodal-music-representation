import os
import sys
import glob
import json

from pydub import AudioSegment


data_home_dir = sys.argv[1]

DOWN_DATA_DIR = os.path.join(data_home_dir, 'yt8m-av-down-data')
CLIPS_DATA_DIR = os.path.join(data_home_dir, 'yt8m-audio-clips')
splits = ['dev', 'test']

threshold = 8000
clip_off_last = 3000
file_format = 'mp3'

for split in splits:
    input_split_dir = os.path.join(DOWN_DATA_DIR, split)
    output_split_dir = os.path.join(CLIPS_DATA_DIR, split)
    info_file_name = f'audio-clips-info-{split}.jsonl'

    os.makedirs(output_split_dir, exist_ok=True)
    os.system(f"touch {CLIPS_DATA_DIR}/{info_file_name}")

    for vid in os.listdir(input_split_dir):
        if len(os.listdir(os.path.join(input_split_dir, vid))) != 2:
            continue
        audio_file = glob.glob(pathname=f'{input_split_dir}/{vid}/{vid}-audio.*')[0]
        audio_segment = AudioSegment.from_file(audio_file)
        dur_ms = len(audio_segment)
        dur_sec = len(audio_segment) / 1000
        start = 0
        end = 0
        c = 0

        os.makedirs(os.path.join(output_split_dir, vid), exist_ok=True)
        while start < (dur_ms - clip_off_last):
            end += threshold
            clip = audio_segment[start:end]
            c += 1
            clip.export(os.path.join(output_split_dir, vid, f'{vid}-audio-{c}.{file_format}'),
                        format=file_format)
            start += threshold
        info = {
            'vid': vid,
            'duration_sec': dur_sec,
            'n_clips': c,
            'last_clip_dur': len(clip) / 1000.0
        }
        with open(os.path.join(CLIPS_DATA_DIR, info_file_name), 'a') as f:
            f.write(json.dumps(info) + '\n')
