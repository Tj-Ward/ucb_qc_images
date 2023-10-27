#!/usr/bin/env bash
export DISPLAY=""
#$ -j y
#$ -o /home/jagust/petcore/qc/scripts/SGE_logs/$JOB_NAME_$HOSTNAME.$JOB_ID
#$ -S /bin/bash
#$ -V
#$ -m a
source $HOME/.bash_profile
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
echo Starting
cd /home/jagust/petcore/qc/scripts

python qc_master_code.py --project "SCAN-FREE" -t="NAV" --ptype="SUVR" -d="/home/jagust/xnat/squid/scan-free" -o="/home/jagust/petcore/qc/scan-free/nav-suvr" -S="NACC\d{4,20}" --mrifree

echo Complete!
