#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts

python run_image_qc_4rtni.py -t="amyloid" -d="/home/jagust/4rtni/data" -o="/home/jagust/petcore/qc/4rtni/amyloid" -S="\d{3}_\d{3,7}" --niftialign

echo Complete!
