#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts

python qc_master_code.py --project "BACS" -t="PIB" --ptype="SUVR" -d="/home/jagust/xnat/squid/bacs" -o="/home/jagust/petcore/qc/bacs/pib-suvr" -S="B\d{1,10}-\d{1,10}" --niftialign


echo Complete!
