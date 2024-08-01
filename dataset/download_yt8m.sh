#!/usr/bin/env bash

# Download the video-level YouTube-8M Dataset
pwd=$PWD
mkdir -p $pwd/data/yt8m/
cd $pwd/data/yt8m/

# If you are located in Europe or Asia, please replace us in the URL with eu or asia,
# respectively to speed up the transfer of the files.

curl data.yt8m.org/download.py | partition=2/video/train mirror=us python
curl data.yt8m.org/download.py | partition=2/video/validate mirror=us python
curl data.yt8m.org/download.py | partition=2/video/test mirror=us python
mv validate dev

cd $pwd

# Download the YouTube-8M vocabulary of entities
wget https://research.google.com/youtube8m/csv/2/vocabulary.csv $pwd/yt8m-vocab.csv
