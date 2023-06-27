#!/bin/bash
#SBATCH -A irel
#SBATCH -c 8
#SBATCH --mem-per-cpu 2G
#SBATCH --gres gpu:0
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/down-yt8m-train-data.log
#SBATCH --mail-user sagar.joshi@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name down-train

mkdir -p /scratch/sagarsj42 && cd $_

scp -r sagarsj42@gnode034:/scratch/sagarsj42/yt8m-filtered-ids .
touch yt8m-av-down-ids.txt

python ~/yt8m/download_av_from_filtered_yt8m.py
