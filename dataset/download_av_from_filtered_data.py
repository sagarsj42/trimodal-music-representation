import os
import sys

from icecream import ic


data_home_dir = sys.argv[1]

FILTERED_DATA_PATH = os.path.join(data_home_dir, 'yt8m-filtered-ids')
DOWN_IDS_PATH = os.path.join(data_home_dir, 'yt8m-av-down-ids.txt')
DOWN_DATA_PATH = os.path.join(data_home_dir, 'yt8m-av-down-data')

open(DOWN_IDS_PATH, 'a').close()

for split in ['train', 'dev', 'test']:
    with open(os.path.join(FILTERED_DATA_PATH, 'split.txt'), 'r') as f:
        split_ids = f.read().split('\n')

    ic(split, len(split_ids))

    os.makedirs(os.path.join(DOWN_DATA_PATH, f'{split}'), exist_ok=True)
    for idx in split_ids:
        with open(DOWN_IDS_PATH, 'r') as f:
            all_down_ids = set(f.read().split('\n'))
        if idx in all_down_ids:
            continue
        down_path = os.path.join(DOWN_DATA_PATH, f'{split}', f'{idx}')
        os.makedirs(down_path, exist_ok=True)
        os.system(f"cd {down_path} && yt-dlp -o '%(id)s-audio.%(ext)s' -f 'ba' -S +size {idx}")
        os.system(f"cd {down_path} && yt-dlp -o '%(id)s-video.%(ext)s' -f 'bv' -S 'res:240,fps' {idx}")
        with open(DOWN_IDS_PATH, 'a') as f:
            f.write(idx + '\n')
