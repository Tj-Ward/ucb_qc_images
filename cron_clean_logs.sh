#!/usr/bin/env bash
#
# These are logs AFNI creates but are useless
# 
# They accumulate when you run AFNI code a lot
# This causes a lot of disk space over time
# Delete them periodically using this script. 
#
# Visual QC code relies on AFNI for straightening the head
#

AFNILOG=$HOME/.afni.log
[ -d $AFNILOG ] && rm $AFNILOG

AFNILOG=$HOME/.afni.crashlog
[ -d $AFNILOG ] && rm $AFNILOG
