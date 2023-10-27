#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts 

python run_image_qc.py -t="FBB" -d="/home/jagust/petcore/pointer/pointer_fbb" -o="/home/jagust/petcore/qc/pointer/fbb" -S="\d{4,25}" --niftialign

echo Complete!
