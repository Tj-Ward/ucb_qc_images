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
# python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -r
# sleep 5
# echo "Temporarily removing spreadsheet from Google while changes are being made"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" --delete
# 
#python qc_master_code.py --project "ADNI" -t="FBB" --ptype="SUVR" -d="/home/jagust/adni/FS7_adni_florbetaben" -o="/home/jagust/petcore/qc/adni-legacy/fbb-suvr" -S="\d{3}-S-\d{4}" --niftialign --legacy_data --inorm
python qc_master_code.py --project "ADNI-DEFACE" -t="FBB" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni" -o="/home/jagust/petcore/qc/adni-deface/fbb-suvr" -S="(000-S-|999-S-)\d{4}" --niftialign
# 
# echo "Updating QC images on Google"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -R
# echo "Uploading files...if new rows are available."
# python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" --newTSV
# echo "Update Needs QC"
# python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -Q

echo Complete!
