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

# echo "Download files...if new file is on google."
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" -r
# sleep 5
# echo "Temporarily removing spreadsheet from Google while changes are being made"
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --delete
# 
python qc_master_code.py --project "UCSF" -t="PIB" --ptype="SUVR" -d="/home/jagust/xnat/squid/ucsf" -o="/home/jagust/petcore/qc/ucsf/pib-suvr" -S="B\d{1,10}-\d{1,10}" --niftialign
#python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" -R
# 
# echo "Updating QC images on Google"
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" -I
# echo "Uploading files...if new rows are available."
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --newTSV
# echo "Update Needs QC"
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" -Q

echo Complete!
