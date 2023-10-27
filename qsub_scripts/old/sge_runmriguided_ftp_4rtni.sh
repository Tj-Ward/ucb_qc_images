#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts
#conda activate neuro

python qc_master_code.py -t="FTP" -d="/home/jagust/4rtni/data" -o="/home/jagust/petcore/qc/4rtni/ftp" -S="\d{3}_\d{3,7}" --niftialign

echo Complete!
