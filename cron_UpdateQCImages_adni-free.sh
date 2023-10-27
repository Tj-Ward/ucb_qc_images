#!/bin/bash
set -e
echo "Starting create QC images..."
source /home/jagust/petcore/qc/scripts/pythonenv/bin/activate
cd /home/jagust/petcore/qc/scripts 

########################
#
# SGE module. 
#  0.) Does not run if the project is busy (Visual QC img creation already running)
#  1.) This file downloads QC TSV off Google
#  2.) Deletes the TSV on Google (so that no changes can be made while creating QC images)
#  3.) Marks SGE as "busy" so that no spreadsheet is uploaded until QC images are created.
#  
# This sucks but SGE processing stations can not interact with the world wide web! 
# Thus, we interact with Google on the workstation, qsub QC images for creation, then the workstation will check to see if QC images are done processing and reupload the TSV, update needsQC, etc in a file that runs nightly.
#
# It goes like this.
#  1.) Download the QC CSV on google (grabs new values reviewers added)
#  2.) Delete the QC CSV on google
#  3.) Send to Google the busy sign (let users know processing is occuring, QC is offline for now)
#  4.) Run QC creation script with the new QC CSV. When done, it deletes the local busy sign. This is important. Allows the next cron file to run. 
# 
#  At the end, the busy sign is still on Google. When the next crontab runs, it will check to see if the local busy sign is gone. If so, it will run and remove the busy sign.
#
#######################


# When adding new projects, copy the HEAD for loop and replace "head" with the new study

# HEAD_TRACERS="FTP-SUVR PIB-SUVR NAV-SUVR FBB-SUVR MK6240-SUVR"
# for TRACER in $HEAD_TRACERS;do
#         echo "HEAD ${TRACER}"
#        BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_head_${TRACER,,}.txt"
#        if [ ! -d "$BUSYDIR" ]; then
#                 python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" -r
#                 sleep 2
#                 python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --delete
#                 python qc_sync_cluster_and_google.py -p "head" -t "${TRACER,,}" --updateBusy "up"
#                #qsub -N head_QC_${TRACER}_$datestring -l mem_free=4 -binding linear:1 -v "QSUB_STUDY=HEAD" -v "QSUB_TRACER=${TRACER}" /home/jagust/petcore/qc/scripts/test.sh
#                 qsub -N head_QC_${TRACER}_$datestring -l mem_free=4 -binding linear:1 -v "QSUB_STUDY=HEAD" -v "QSUB_TRACER=${TRACER}" /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_generic.sh
#         fi
# done


# JQ: Subject ID uses underscore instead of dash. Synced images are for the defacing validation test set (to be removed when adni-free is shared with UCSF)
# JQ: Remove FTP
# ADNIFREE_TRACERS="FTP"
ADNIFREE_TRACERS="FBP FBB"
for TRACER in $ADNIFREE_TRACERS;do
        echo "ADNI-FREE"
        BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_adni-free_${TRACER,,}-suvr.txt"
        if [ ! -d "$BUSYDIR" ]; then
                python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --createFolders
                python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" -r
                sleep 2
                #python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --delete
                python qc_sync_cluster_and_google.py -p "adni-free" -t "${TRACER,,}-suvr" --updateBusy "up"
                qsub -N adni-free_QC_${TRACER}-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_${TRACER,,}-suvr_adni-free.sh
        fi
done

# #  Below is the ADNI Deface stuff
# # 
# ADNI_TRACERS="FBP FBB FTP"
# for TRACER in $ADNI_TRACERS;do
#         echo "ADNI-DEFACE"
#         BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_adni-deface_${TRACER,,}-suvr.txt"
#         if [ ! -d "$BUSYDIR" ]; then
# 		qsub -N adni-deface_QC_${TRACER}-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_${TRACER,,}-suvr_adni-deface.sh
#         fi
# done


# echo "ADNI"
# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_adni_fbb-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
# 	# Download reviews
#         python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" -r
#         sleep 2
#         python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" --delete
#         python qc_sync_cluster_and_google.py -p "adni" -t "fbb-suvr" --updateBusy "up"
# 	# Qsub will create QC images, update spreadsheet against squid, and the upload the spreadsheet to Google
#         qsub -N adni_QC_FBB-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_fbb-suvr_adni.sh
# fi


# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_adni_ftp-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
# 	echo "Download files...if new file is on google."
# 	python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" -r
# 	sleep 2
# 	echo "Removing QC TSV from Google"
# 	python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" --delete
# 	python qc_sync_cluster_and_google.py -p "adni" -t "ftp-suvr" --updateBusy "up"
# 	qsub -N adni_QC_FTP-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_ftp-suvr_adni.sh
# fi

# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_adni_fbp-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
# 	python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" -r
# 	sleep 2
# 	python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" --delete
# 	python qc_sync_cluster_and_google.py -p "adni" -t "fbp-suvr" --updateBusy "up"
# 	qsub -N adni_QC_FBP-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_fbp-suvr_adni.sh
# fi


# # BACS Download, Upload, Sync imgs, Needs QC
# echo "BACS"
# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_bacs_ftp-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
# 	echo "Download files...if new file is on google."
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" -r
# 	sleep 2
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --delete
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "ftp-suvr" --updateBusy "up"
# 	qsub -N bacs_QC_FTP-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_ftp-suvr_bacs.sh
# fi

# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_bacs_pib-dvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" -r
# 	sleep 2
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --delete
# 	python qc_sync_cluster_and_google.py -p "bacs" -t "pib-dvr" --updateBusy "up"
# 	qsub -N bacs_QC_PIB-DVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_pib-dvr_bacs.sh
# fi
# #
# #BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_bacs_pib-suvr.txt"
# #if [ ! -d "$BUSYDIR" ]; then
# #	#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" -r
# #	sleep 2
# #	#python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --delete
# #	python qc_sync_cluster_and_google.py -p "bacs" -t "pib-suvr" --updateBusy "up"
# #	qsub -N bacs_QC_PIB-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_pib-suvr_bacs.sh
# #fi
# #
# #echo "UCSF"
# ## UCSF
# #BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_ucsf_pib-suvr.txt"
# #if [ ! -d "$BUSYDIR" ]; then
# #        #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" -r
# #        sleep 2
# #        #python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --delete
# #        python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-suvr" --updateBusy "up"
# #        qsub -N ucsf_QC_PIB-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_pib-suvr_ucsf.sh
# #fi
# #
# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_ucsf_pib-dvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" -r
#         sleep 2
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --delete
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "pib-dvr" --updateBusy "up"
#         qsub -N ucsf_QC_PIB-DVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_pib-dvr_ucsf.sh
# fi

# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_ucsf_ftp-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" -r
#         sleep 2
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --delete
#         python qc_sync_cluster_and_google.py -p "ucsf" -t "ftp-suvr" --updateBusy "up"
#         qsub -N ucsf_QC_FTP-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_ftp-suvr_ucsf.sh
# fi


# echo "POINTER"
# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_pointer_mk6240-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
#         python qc_sync_cluster_and_google.py -p "pointer" -t "mk6240-suvr" -r
#         sleep 2
#         python qc_sync_cluster_and_google.py -p "pointer" -t "mk6240-suvr" --delete
#         python qc_sync_cluster_and_google.py -p "pointer" -t "mk6240-suvr" --updateBusy "up"
#         qsub -N pointer_QC_MK6240-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_mk6240-suvr_pointer.sh
# fi

# BUSYDIR="/home/jagust/petcore/qc/scripts/required/BusySign_pointer_fbb-suvr.txt"
# if [ ! -d "$BUSYDIR" ]; then
#         python qc_sync_cluster_and_google.py -p "pointer" -t "fbb-suvr" -r
#         sleep 2
#         python qc_sync_cluster_and_google.py -p "pointer" -t "fbb-suvr" --delete
#         python qc_sync_cluster_and_google.py -p "pointer" -t "fbb-suvr" --updateBusy "up"
#         qsub -N pointer_QC_FBB-SUVR_$datestring -l mem_free=4 -binding linear:1 /home/jagust/petcore/qc/scripts/qsub_scripts/sge_runmriguided_fbb-suvr_pointer.sh
# fi
# echo "Who is QCing png image"
# sh who_is_qcing.sh
# exit 0
