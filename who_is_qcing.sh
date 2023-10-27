#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
cd /home/jagust/petcore/qc/scripts


python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --downloadreviews
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "pointer" -t "mk6240-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "pointer" -t "fbb-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --downloadreviews

# Creates the auto-plots
# right now, just who is qcing
python who_is_qcing.py

# Sync autoplot to neurocluster
python qc_sync_cluster_and_google.py --autoPlots

