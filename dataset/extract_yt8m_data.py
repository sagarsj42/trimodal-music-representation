import os
import sys
import json
import warnings

import requests
import pandas as pd
import tensorflow as tf


def extract_video_info(tf_record_path, label_yt8m_indices):
    video_infos = list()
    for raw_rec in tf.data.TFRecordDataset(tf_record_path):
        example = tf.train.Example()
        example.ParseFromString(raw_rec.numpy())

        labels = [v for v in example.features.feature['labels'].int64_list.value]
        if len(set(labels).intersection(set(label_yt8m_indices.keys()))) < 1:
            continue
        label_values = list()
        for label in labels:
            if label in label_yt8m_indices:
                label_values.append(label_yt8m_indices[label])

        data_id = example.features.feature['id'].bytes_list.value[0].decode()
        try:
            prefix = data_id[:2]
            vid_req_url = f'https://data.yt8m.org/2/j/i/{prefix}/{data_id}.js'
            response = requests.get(vid_req_url, verify=False)
            vid = response.text.split(',')[1].split('"')[1]
        except:
            continue

        mean_rgb = [v for v in example.features.feature['mean_rgb'].float_list.value]
        mean_audio = [v for v in example.features.feature['mean_audio'].float_list.value]

        video_infos.append({
            'vid': vid,
            'labels': label_values,
            'mean_rgb': mean_rgb,
            'mean_audio': mean_audio
        })

    return video_infos


data_home_dir = sys.argv[1]
splits = sys.argv[2].split(',')

data_dir = os.path.join(data_home_dir, 'yt8m')
yt8m_vocab_file = os.path.join(data_home_dir, 'yt8m-vocab.csv')
select_labels_list = os.path.join(data_home_dir, 'select-music-labels.txt')
extracted_labels_data_folder = os.path.join(data_home_dir, 'yt8m-label-extracted')
extracted_features_data_folder = os.path.join(data_home_dir, 'yt8m-features-extracted')

warnings.filterwarnings('ignore')
with open(select_labels_list, 'r') as f:
    labels = f.read().split('\n')

yt8m_vocab_df = pd.read_csv(yt8m_vocab_file)

print(yt8m_vocab_df.info())

label_yt8m_indices = dict()
for label in labels:
    yt8m_indx = yt8m_vocab_df[yt8m_vocab_df['Name'] == label].iloc[0]['Index']
    label_yt8m_indices[yt8m_indx] = label

for split in splits:
    for i, tf_record_filename in enumerate(os.listdir(os.path.join(data_dir, split))):
        if i % 100 == 0:
            print(f'Processing {split} record {i}')
        extracted_data = extract_video_info(os.path.join(data_dir, split, tf_record_filename),
                                            label_yt8m_indices)

        label_data = list()
        os.makedirs(extracted_labels_data_folder, exist_ok=True)
        for info in extracted_data:
            label_data.append({
                'vid': info['vid'],
                'labels': info['labels']
            })
        label_data_jsonl = '\n'.join([json.dumps(d) for d in label_data]) + '\n'
        with open(os.path.join(extracted_labels_data_folder, f'{split}.jsonl'), 'a') as f:
            f.write(label_data_jsonl)

        os.makedirs(os.path.join(extracted_features_data_folder, split), exist_ok=True)
        for info in extracted_data:
            vid = info['vid']
            with open(os.path.join(extracted_features_data_folder, split, f'{vid}.json'), 'w') as f:
                json.dump(info, f)
