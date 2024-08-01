#!/usr/bin/env bash

DATA_HOME_DIR=../data/

# To run everything using only the dev set:
SPLITS=dev

# To use the entire dataset:
#SPLITS=train,dev,test

bash ./download_yt8m.sh

python3 ./extract_yt8m_data.py $DATA_HOME_DIR $SPLITS

python3 ./download_video_metadata.py $DATA_HOME_DIR $SPLITS

python3 ./filter_videos_from_metadata.py $DATA_HOME_DIR $SPLITS

python3 ./download_av_from_filtered_data.py $DATA_HOME_DIR

python3 ./split_av_into_sparse_clips.py $DATA_HOME_DIR

python3 ./split_audio_into_clips.py $DATA_HOME_DIR

python3 ./split_video_into_clips.py $DATA_HOME_DIR
