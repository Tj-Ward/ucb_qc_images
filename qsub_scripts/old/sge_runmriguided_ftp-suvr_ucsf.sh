#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts

python qc_master_code.py --project "UCSF" --tracer="FTP" --ptype="SUVR" -d="/home/jagust/xnat/squid/ucsf" -o="/home/jagust/petcore/qc/ucsf/ftp-suvr" -S="B\d{1,5}-\d{1,10}"  --niftialign

echo Complete!
