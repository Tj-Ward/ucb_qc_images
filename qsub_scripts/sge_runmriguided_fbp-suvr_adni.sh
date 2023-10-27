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
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -r
# sleep 5
# echo "Temporarily removing spreadsheet from Google while changes are being made"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" --delete
# 
#python qc_master_code.py --project "ADNI" -t="FBP" --ptype="SUVR" -d="/home/jagust/adni/FS7_adni_av45" -o="/home/jagust/petcore/qc/adni-legacy/fbp-suvr" -S="\d{3}-S-\d{4}" --niftialign --legacy_data --inorm
python qc_master_code.py --project "ADNI" -t="FBP" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni" -o="/home/jagust/petcore/qc/adni/fbp-suvr" -S="(?!000-S-|999-S-)\d{3}-S-\d{4}" --niftialign
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -R
# 
# echo "Updating QC images on Google"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -I
# echo "Uploading files...if new rows are available."
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" --newTSV
# echo "Update Needs QC"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -Q

echo Complete!
