import os,re,glob
import pandas as pd
from datetime import datetime
import subprocess
FILE_ = '../adni/ftp-suvr/FLAGGED_QC_ADNI_FTP-SUVR_05-01-2023.tsv'
TMP_PATH='/home/jagust/tjward/NotBackedUp/tmp_flagged_imgs/'

df = pd.read_csv(FILE_,header=0,sep='\t')

for index, row in df.iterrows():
    date = row['PET Date']
    subject = row['ID']
    imgs=glob.glob(f'../adni/ftp-suvr/v*/*{subject}*{date}*.png')
    if len(imgs) == 1:
        subprocess.run(['cp',imgs[0],TMP_PATH])

    else:
        print(imgs)
