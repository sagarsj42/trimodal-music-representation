#!/bin/bash
#SBATCH -A irel
#SBATCH -c 8
#SBATCH --mem-per-cpu 2G
#SBATCH --gres gpu:1
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/extract-yt8m-dev-test-data.log
#SBATCH --mail-user sagar.joshi@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name dt-ext-yt

mkdir -p /scratch/sagarsj42 && cd $_

scp -r sagarsj42@gnode034:/scratch/sagarsj42/yt8m .
scp -r sagarsj42@gnode034:/scratch/sagarsj42/yt8m-vocab.csv .
scp -r sagarsj42@gnode034:/scratch/sagarsj42/select-music-labels.txt .

python ~/yt8m/extract_data_from_yt8m.py
