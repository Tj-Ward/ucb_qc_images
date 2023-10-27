#!/usr/bin/env bash
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate

echo "Starting regular sync"

cd /home/jagust/petcore/qc/scripts 

########################
#
# Local module. Intended to be ran often. 
#  1.) Does not run if BusySign is up
#  2.) 
#  
# This sucks but SGE processing stations can not interact with the internet! 
# Thus, we interact with Google on the workstation, qsub QC images for creation, then the workstation will check to see if QC images are done processing and reupload the TSV, update needsQC, etc in a file that runs nightly.
#
#######################

# Download, Upload, Sync imgs, Needs QC
# Only uploads TSV if not on Google already.

#HEAD_TRACERS="FTP-SUVR PIB-SUVR NAV-SUVR FBB-SUVR MK6240-SUVR"
#for TRACER in $HEAD_TRACERS;do
#	echo "HEAD ${TRACER}"
#	python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --mrifree --uploadPNGs
#        python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --mrifree --newTSV # Does nothing if a TSV is already on GOOGLE
#        python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --mrifree --downloadreviews
#        python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --mrifree --needsQC # A google TSV is needed to run needs QC
#        python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --mrifree --updateBusy "down"
#done

#ADNIFREE_TRACERS="FBP FBB FTP"
#for TRACER in $ADNIFREE_TRACERS;do
#        echo "ADNI-FREE ${TRACER}"
#	python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --mrifree --uploadPNGs
#	python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --mrifree --newTSV # Does nothing if a TSV is already on GOOGLE
#	python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --mrifree --downloadreviews
#	python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --mrifree --needsQC # A google TSV is needed to run needs QC
#	python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --mrifree --updateBusy "down"
#done
# 
# SCAN_TRACERS="FBP FBB PIB NAV FTP MK6240"
# for TRACER in $SCAN_TRACERS;do
#         echo "SCAN ${TRACER}"
# 	python qc_sync_cluster_and_google.py -p "scan-free" -t "${TRACER,,}-suvr" --mrifree --uploadPNGs
# 	python qc_sync_cluster_and_google.py -p "scan-free" -t "${TRACER,,}-suvr" --mrifree --newTSV # Does nothing if a TSV is already on GOOGLE
# 	python qc_sync_cluster_and_google.py -p "scan-free" -t "${TRACER,,}-suvr" --mrifree --downloadreviews
# 	# python qc_sync_cluster_and_google.py -p "scan-free" -t "${TRACER,,}-suvr" --mrifree --needsQC # A google TSV is needed to run needs QC
# 	python qc_sync_cluster_and_google.py -p "scan-free" -t "${TRACER,,}-suvr" --mrifree --updateBusy "down"
# done
# 
# 
POINTER_TRACERS="FBB MK6240"
for TRACER in $POINTER_TRACERS;do
        echo "POINTER ${TRACER}"
        python qc_sync_cluster_and_google.py -p "pointer" -t "${TRACER,,}-suvr" --uploadPNGs
        python qc_sync_cluster_and_google.py -p "pointer" -t "${TRACER,,}-suvr" --newTSV
        python qc_sync_cluster_and_google.py -p "pointer" -t "${TRACER,,}-suvr" --downloadreviews
        python qc_sync_cluster_and_google.py -p "pointer" -t "${TRACER,,}-suvr" --needsQC
        python qc_sync_cluster_and_google.py -p "pointer" -t "${TRACER,,}-suvr" --updateBusy "down"
done


ADNI_TRACERS="FBB FTP FBP"
for TRACER in $ADNI_TRACERS;do
	echo  ADNI "${TRACER}"
	python qc_sync_cluster_and_google.py -p "adni" -t "${TRACER,,}-suvr" --uploadPNGs
	python qc_sync_cluster_and_google.py -p "adni" -t "${TRACER,,}-suvr" --newTSV # Does nothing if a TSV is already on GOOGLE
	python qc_sync_cluster_and_google.py -p "adni" -t "${TRACER,,}-suvr" --downloadreviews
	python qc_sync_cluster_and_google.py -p "adni" -t "${TRACER,,}-suvr" --needsQC # A google TSV is needed to run needs QC
	python qc_sync_cluster_and_google.py -p "adni" -t "${TRACER,,}-suvr" --updateBusy "down"
done

echo BACS
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --uploadPNGs
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --newTSV
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --needsQC
python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --updateBusy "down"

python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --uploadPNGs
python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --newTSV
python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --downloadreviews
python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --needsQC
python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --updateBusy "down"

#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --uploadPNGs
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --newTSV
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --downloadreviews
# We are not planning to QC all of bacs SUVRs so the needs QC folder is empty but everything else is there just for the hell of it
# python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --needsQC
#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --updateBusy "down"

# echo UCSF
# python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --uploadPNGs
# python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --newTSV
# python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --downloadreviews
# python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --needsQC
# python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --updateBusy "down"
# 
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --uploadPNGs
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --newTSV
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --downloadreviews
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --needsQC
# python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --updateBusy "down"
# 
# #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --uploadPNGs
# #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --newTSV
# #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --downloadreviews
# #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --needsQC
# #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --updateBusy "down"
# 
# 
 
echo "Who is QCing png image"
bash who_is_qcing.sh
