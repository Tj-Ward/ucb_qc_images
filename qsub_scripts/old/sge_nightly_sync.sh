#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
cd /home/jagust/petcore/qc/scripts 


##### BACS
echo "Starting generation... BACS!"

echo "Download new reviews..."
python qc_sync_cluster_and_google.py -p "bacs" --tracer "ftp-suvr" --downloadreviews
#python qc_sync_cluster_and_google.py -p "bacs" --tracer "pib-dvr" --downloadreviews
#python qc_sync_cluster_and_google.py -p "bacs" --tracer "pib-suvr" --downloadreviews
sleep 2
python qc_spreadsheet_creation.py -p "bacs" --tracer "ftp-suvr"
#python qc_spreadsheet_creation.py -p "bacs" --tracer "pib-dvr"
#python qc_spreadsheet_creation.py -p "bacs" --tracer "pib-suvr"
sleep 1
echo "Upload new spreadsheets..."
python qc_sync_cluster_and_google.py -p "bacs" --tracer "ftp-suvr" --uploadrows
#python qc_sync_cluster_and_google.py -p "bacs" --tracer "pib-dvr" --uploadrows
#python qc_sync_cluster_and_google.py -p "bacs" --tracer "pib-suvr" --uploadrows

echo "Upload PNGs...if new pngs are on cluster."
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" -I
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" -I
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" -I

echo "Refreshing needs qc folder...if necessary"
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" -Q
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" -Q
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" -Q

### UCSF
echo "Starting generation... UCSF!"

echo "Download new reviews..."
python qc_sync_cluster_and_google.py -p "ucsf" --tracer "ftp-suvr" --downloadreviews
sleep 2
python qc_spreadsheet_creation.py -p "ucsf" --tracer "ftp-suvr"
sleep 1
echo "Upload new spreadsheets..."
python qc_sync_cluster_and_google.py -p "ucsf" --tracer "ftp-suvr" --uploadrows

echo "Upload PNGs...if new pngs are on cluster."
python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" -I

echo "Refreshing needs qc folder...if necessary"
python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" -Q


##### ADNI
echo "Starting generation... ADNI!"

echo "Download new reviews..."
python qc_sync_cluster_and_google.py -p "adni" --tracer "ftp-suvr" --downloadreviews
#python qc_sync_cluster_and_google.py -p "adni" --tracer "fbp-suvr" --downloadreviews
#python qc_sync_cluster_and_google.py -p "adni" --tracer "fbb-suvr" --downloadreviews
sleep 2
echo "Create new rows of spreadsheet..."
python qc_spreadsheet_creation.py -p "adni" --tracer "ftp-suvr"
#python qc_spreadsheet_creation.py -p "adni" --tracer "fbp-suvr"
#python qc_spreadsheet_creation.py -p "adni" --tracer "fbb-suvr"
sleep 1
echo "Upload new spreadsheets..."
python qc_sync_cluster_and_google.py -p "adni" --tracer "ftp-suvr" --uploadrows
#python qc_sync_cluster_and_google.py -p "adni" --tracer "fbp-suvr" --uploadrows
#python qc_sync_cluster_and_google.py -p "adni" --tracer "fbb-suvr" --uploadrows

echo "Upload PNGs...if new pngs are on cluster."
python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -I
#python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -I
#python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -I

echo "Refreshing needs qc folder...if necessary"
python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -Q
#python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -Q
#python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -Q





#echo "Starting generation... POINTER!"
#
#echo "Download new reviews..."
#python qc_sync_cluster_and_google.py -p "pointer" --downloadreviews
#if [ $? -eq 0 ]; then
#    echo QC SYNC CLUSTER GOOGLE - OK
#else
#    echo FAIL
#    exit 1
#fi
#sleep 10
#echo "Create new rows of spreadsheet..."
#python qc_spreadsheet_creation.py -p "pointer" 
#if [ $? -eq 0 ]; then
#    echo QC SPREADSHEED CREATION - OK
#else
#    echo FAIL
#    exit 1
#fi
#sleep 10
#echo "Upload new spreadsheets..."
#python qc_sync_cluster_and_google.py -p "pointer" --uploadrows
#if [ $? -eq 0 ]; then
#    echo QC SYNC CLUSTER GOOGLE - OK
#else
#    echo FAIL
#    exit 1
#fi
