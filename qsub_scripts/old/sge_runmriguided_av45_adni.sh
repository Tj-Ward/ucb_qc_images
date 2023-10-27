#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts

python qc_master_code.py --project "ADNI" -t="FBP" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni" -o="/home/jagust/petcore/qc/adni/fbp-suvr" -S="\d{3}-S-\d{4}"  --niftialign

echo Complete!
