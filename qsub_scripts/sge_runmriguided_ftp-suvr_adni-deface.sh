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
# python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -r
# sleep 5
# echo "Temporarily removing spreadsheet from Google while changes are being made"
# python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" --delete
# 
# #python qc_master_code.py --project "ADNI" -t="FTP" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni" -o="/home/jagust/petcore/qc/adni/ftp-suvr" -S="B\d{1,10}-\d{1,10}" --niftialign
# 
#python qc_master_code.py --project "ADNI" -t="FTP" --ptype="SUVR" -d="/home/jagust/adni/FS7_adni_av1451" -o="/home/jagust/petcore/qc/adni-legacy/ftp-suvr" -S="\d{3}-S-\d{4}" --niftialign --legacy_data --inorm
python qc_master_code.py --project "ADNI-DEFACE" -t="FTP" --ptype="SUVR" -d="/home/jagust/xnat/squid/adni" -o="/home/jagust/petcore/qc/adni-deface/ftp-suvr" -S="(000-S-|999-S-)\d{4}" --niftialign
#python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -R
# 
# echo "Updating QC images on Google"
# python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -I
# echo "Uploading files...if new rows are available."
# python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" --newTSV
# echo "Update Needs QC"
# python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -Q

echo Complete!
