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

python qc_master_code.py --project "ADNI-FREE" -t="FTP" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni-free" -o="/home/jagust/petcore/qc/adni-free/ftp-suvr" -S=".*" --mrifree

echo Complete!
