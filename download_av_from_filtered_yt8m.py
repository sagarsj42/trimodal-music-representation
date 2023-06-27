import os
os.chdir('/scratch/sagarsj42')

import subprocess
from icecream import ic


FILTERED_DATA_PATH = './yt8m-filtered-ids'
DOWN_IDS_PATH = './yt8m-av-down-ids.txt'
DOWN_DATA_PATH = './yt8m-av-down-data'

with open(os.path.join(FILTERED_DATA_PATH, 'train.txt'), 'r') as f:
    train_ids = f.read().split('\n')

with open(os.path.join(FILTERED_DATA_PATH, 'dev.txt'), 'r') as f:
    dev_ids = f.read().split('\n')

with open(os.path.join(FILTERED_DATA_PATH, 'test.txt'), 'r') as f:
    test_ids = f.read().split('\n')

ic(len(train_ids), len(dev_ids), len(test_ids))

split = 'train'
os.makedirs(os.path.join(DOWN_DATA_PATH, f'{split}'), exist_ok=True)
for idx in train_ids:
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
