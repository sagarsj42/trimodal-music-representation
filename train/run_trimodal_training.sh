#!/bin/bash
#SBATCH -A research
#SBATCH -c 9
#SBATCH -w gnode060
#SBATCH --mem-per-cpu 2G
#SBATCH --gres gpu:1
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/trimodal-contrastive.log
#SBATCH --mail-user sagar.joshi@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name cont

mkdir -p /scratch/sagarsj42 && cd $_

# scp -r sagarsj42@ada:/share1/sagarsj42/yt8m-clips-dataset-info .
# scp sagarsj42@ada:/share1/sagarsj42/yt8m-video-clips-dev-to-train2.zip .
# scp sagarsj42@ada:/share1/sagarsj42/yt8m-video-clips-train1345.zip .
# scp sagarsj42@ada:/share1/sagarsj42/yt8m-audio-clips-dev-to-train2.zip .
# scp sagarsj42@ada:/share1/sagarsj42/yt8m-audio-clips-train1345.zip .

# unzip -q yt8m-video-clips-dev-to-train2.zip
# rm -rf yt8m-video-clips/train1
# rm -rf yt8m-video-clips/video-clips-info-train1.jsonl
# unzip -q yt8m-video-clips-train1345.zip

# unzip -q yt8m-audio-clips-dev-to-train2.zip
# rm -rf yt8m-audio-clips/train1
# rm -rf yt8m-audio-clips/audio-clips-info-train1.jsonl
# unzip -q yt8m-audio-clips-train1345.zip

# wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt

# python ~/yt8m/preprocess_av_clips.py

# python ~/yt8m/train_cosine_sim_dissim.py
# python ~/yt8m/train_weighted_cosine_sim_dissim.py
python ~/yt8m/train_contrastive.py
# python ~/yt8m/train_weighted_contrastive.py
