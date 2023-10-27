#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts 

python run_image_qc.py -t="MK6240" -d="/home/jagust/adni/pointer_mk6240" --petpath="/home/jagust/adni/pointer_mk6240_6mm" -o="/home/jagust/petcore/qc/pointer/mk6240_6mm" -S="\d{4,20}" --niftialign

echo Complete!
