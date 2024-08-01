import os
import re
import sys
import math
import datetime

import pandas as pd


data_home_dir = sys.argv[1]
splits = sys.argv[2].split(',')

min_duration_sec = 100
max_duration_sec = 512
min_age_years = 0
min_view_count = 10000
min_like_count = 100
min_comment_count = 5

INFO_DATA_PATH = os.path.join(data_home_dir, 'yt8m-info')
FILTERED_DATA_PATH = os.path.join(data_home_dir, 'yt8m-filtered-ids')

info_df = None

for split in splits:
    split_info_df = pd.read_json(os.path.join(INFO_DATA_PATH, f'{split}.jsonl'), lines=True)
    if info_df:
        info_df = pd.concat([info_df, split_info_df], ignore_index=True)
    else:
        info_df = split_info_df

duration_strs = info_df['duration'].tolist()
duration_secs = list()
for ds in duration_strs:
    if not ds or type(ds) != str:
        duration_secs.append(-1)
        continue

    ds_vals = ds.split(':')
    ds_sec = 0
    for i in range(1, len(ds_vals) + 1):
        ds_sec += (int(ds_vals[-i]) * math.pow(60, i - 1))
    duration_secs.append(ds_sec)
info_df['duration_sec'] = duration_secs

relative_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
published_times = info_df['publish_time'].tolist()
age_years = list()
for pt in published_times:
    curr_date = pt.to_pydatetime().replace(tzinfo=None)
    if type(curr_date) != datetime.datetime:
        age_years.append(-1)
        continue
    curr_sec = (relative_date - curr_date).total_seconds()
    age_years.append(curr_sec / (3600 * 24.0 * 365))
info_df['age_years'] = age_years

titles = info_df['title'].tolist()
title_lens = list()
for t in titles:
    if type(t) != str:
        title_lens.append(-1)
        continue
    else:
        title_lens.append(len(t.split()))
info_df['title_len'] = title_lens

descs = info_df['description'].tolist()
desc_lens = list()
for d in descs:
    if type(d) != str:
        desc_lens.append(-1)
        continue
    d = re.sub(r'\s\s+', ' ', d)
    d = re.sub(r'\n', ' ', d)
    desc_lens.append(len(d.split()))
info_df['desc_len'] = desc_lens

tags = info_df['tags'].tolist()
n_tags = list()
for t in tags:
    if type(t) != str:
        n_tags.append(-1)
        continue
    n_tags.append(len(t.split(',')))
info_df['n_tags'] = n_tags

info_df = info_df[info_df['duration_sec'] >= min_duration_sec]
info_df = info_df[info_df['duration_sec'] <= max_duration_sec]
info_df = info_df[info_df['age_years'] >= min_age_years]
info_df = info_df[info_df['view_count'] >= min_view_count]
info_df = info_df[info_df['like_count'] >= min_like_count]
info_df = info_df[info_df['comment_count'] >= min_comment_count]

test_df = info_df.query('age_years > 7 and age_years <= 8')
dev_df = info_df.query('age_years > 8 and age_years <= 8.5')
train_df = info_df.query('age_years > 8.5')

os.makedirs(FILTERED_DATA_PATH, exist_ok=True)
with open(os.path.join(FILTERED_DATA_PATH, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_df['vid'].tolist()))
with open(os.path.join(FILTERED_DATA_PATH, 'dev.txt'), 'w') as f:
    f.write('\n'.join(dev_df['vid'].tolist()))
with open(os.path.join(FILTERED_DATA_PATH, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_df['vid'].tolist()))
