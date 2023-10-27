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

# This accepts 2 arguments
# example: ADNI FTP-SUVR

STUDY=$QSUB_STUDY
TRACER=${QSUB_TRACER%-*}
PTYPE=${QSUB_TRACER##*-}

python qc_master_code.py --project "$STUDY" -t="$TRACER" --ptype="$PTYPE" -d="/home/jagust/xnat/squid/${STUDY,,}" -o="/home/jagust/petcore/qc/${STUDY,,}/${QSUB_TRACER,,}" -S=".*" --niftialign

echo Complete!
