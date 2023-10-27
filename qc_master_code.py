#!/usr/bin/env python

####
#
# Drawing QC images.
#
# All new projects should use the UCB_figure_drawing functions to draw QC images.
#   - ucbqc.draw_mrifree
#   - ucbqc.draw_mridependent
#
# To create custom QC images, just copy one of those functions and modify it to your will
# At minimum, an intermediate understanding of numpy arrays is recommended
#
####

# inorm=True/False
# inorm variable determines if the PET image will be displayed with or without intensity normalization.
# if inorm is False, the reference dataframe can be empty but the function still expects a reference dataframe so it must be initalized above. 

# I recommend lossless PNG format but any PIL (python pillow package) supported image format will work.


# Lastly, I'll just define the path of the actual coregistered pet image
# nifti and nifti compressed (nii.gz) should both work but I've only tested on nifti right nowi
# mgz would likely be trivial to get working too as mgz is supported by nibabel package


# Consider switch meta file from CSV to parquet


import argparse
import sys
import os,time
import subprocess
import string,random
import shutil

import numpy as np
import nibabel as nib
import glob
import pandas as pd
import re
import csv
from datetime import datetime
from datetime import date as Date
import traceback

from PIL import Image,ImageOps,ImageFont,ImageDraw
from PIL.PngImagePlugin import PngInfo

# Draw image is my image QC generation function
sys.path.append('/home/jagust/petcore/qc/scripts')
import draw_func as di
import pathlib

import UCB_figure_drawing as ucbqc

from automated_flags import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tracer",type=str, help="PET Type (FTP,FBP,FBB,PIB,PI2620,MK6240)", required=True)
parser.add_argument("--ptype",type=str, help="PET Type (SUVR, DVR, ?)", required=True)
parser.add_argument("--project",type=str, help="BACS,ADNI,POINTER, etc..", required=True)
parser.add_argument("-d", "--datapath",type=str, help="FS Data Folder (Ex: /home/jagust/adni/adni_florbetaben/)", required=True)
parser.add_argument("-o", "--outpath",type=str, help="QC Image Folder (Ex: /home/jagust/adni/adni_qc/visual_qc)", required=True)
parser.add_argument("-S", "--subject_match",type=str, help="Subject name or regex (Ex: 128-S-4586, B10-014, B\d{2}-\d{3})", default='\d{3}-S-\d{4}')
parser.add_argument('--niftialign', action='store_true',help="Use nifti_align to straighted images? Default no.",default=False)
parser.add_argument('--inorm', action='store_true',help="Intensity normalize? Default no.",default=False)
parser.add_argument('--mrifree', action='store_true',help="MRI-free pipeline? Default no.",default=False)
parser.add_argument('--removelostdata', action='store_true',help="Remove QC images if data no longer exists? Default no.",default=False)

args = parser.parse_args()

#convert_names = {'AV1451':'FTP',
#                'AV45':'FBP'}

PET_TYPES=['FTP','MK6240','NAV','PI2620','FBB','FBP','PIB']
ABETA_WCEREB = ['FBB','FBP']
ABETA_CEREB_GM = ['PIB','NAV']
TAU_TRACERS = ['MK6240','FTP','PI2620']

pet_tracer = args.tracer
project_name = args.project
ptype = args.ptype
data_path = args.datapath
sub_math = args.subject_match
output_path = args.outpath
remove_lost_data = args.removelostdata
align_QC_img = args.niftialign
inorm=args.inorm
mrifree = args.mrifree
template_aparc = '/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/ADNI200_DK_Atlas/rADNICN200_aparc+aseg_smoo1.50_normalized.nii'


if mrifree:
    QC_columns = ['ID','TP','PET Date','Image Notes','QC Notes','Reviewer Initials','Review Date', 'Usability','Planned Intervention','RA Initials','Reviewed QC Flag','PET Cerebellum Overlap', 'PET Cerebrum Overlap']
    QC_columns_noindex = QC_columns[3:]
    QC_index_cols = ['ID','TP','PET Date']
else:
    QC_columns = ['ID','TP','PET Date','MRI Date','Image Notes','QC Notes','Reviewer Initials','Review Date', 'Usability','Planned Intervention','RA Initials','Reviewed QC Flag','L/R Symmetry','PET Cerebellum Overlap', 'PET Cerebrum Overlap','Longitudinal MRI Alignment']
    QC_columns_noindex = QC_columns[4:]
    QC_index_cols = ['ID','TP','PET Date','MRI Date']

#if pet_tracer in convert_names.keys():
#    pet_tracer = convert_names[pet_tracer]
if pet_tracer not in PET_TYPES:
    raise ValueError(f"Invalid PET type: {' '.join(PET_TYPES)}")

def main():
    '''
    This code creates QC images for preprocessed PET images and their registered MRI / freesurfer segmentation.

    This code runs for each project,tracer combo on SQUID.

    A QC spreadsheet is also created and managed in this code. Each row corresponds to a QC image.

    SQUID data format: <project>/<subjects>/<data>
    
    pet type: tracer abbreviation must be specific. New PET types can be easily added.

    ptype: processing type <suvr/dvr>

    '''

    print('Running visual QC')
    print('PET type:', args.tracer)
    print('Data folder:', args.datapath)
    print('QC image output folder:', args.outpath)
    print('Running for subjects: ', args.subject_match)
    print('Image type:',args.ptype)
    print('i-norm images:',args.inorm)
    print('Nifti align:',args.niftialign)
    print('MRI-free:',args.mrifree)
    
    #if checkIsBusy(project_name.lower(),f'{pet_tracer.lower()}-{ptype.lower()}'):
    #    print('QC busy - Not starting QC image script.')
    #    sys.exit("QC apparently busy")

    print('Updating IS BUSY UP on Google')
    updateIsBusy(project_name.lower(),f'{pet_tracer.lower()}-{ptype.lower()}','up')
    
    print('Running RUN_QC()')
    try: run_qc(project_name=project_name,pet_tracer=pet_tracer,ptype=ptype,data_path=data_path,sub_match=sub_math, output_path=output_path,remove_lost_data=remove_lost_data,align_QC_img=align_QC_img, inorm=inorm, mrifree=mrifree)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print('run_qc failed for unknown reason. See exception above. ')

    print('Updating IS BUSY DOWN')
    
    updateIsBusy(project_name.lower(),f'{pet_tracer.lower()}-{ptype.lower()}','down')

def pettype2name(ptype,tracer,mrifree=False):
    '''
    What is the name of the PET file?
    '''

    if not mrifree:
        CEREB_GM_TRACERS = ['pib','ftp','mk6240','pi2620']
        if np.isin(tracer,CEREB_GM_TRACERS):
            return f'{ptype.lower()}_cereg.nii'

        CEREB_WHOLE_TRACERS = ['fbp','fbb']
        if np.isin(tracer,CEREB_WHOLE_TRACERS):
            return f'{ptype.lower()}_cere.nii'
        else:
            raise ValueError('Error with PET processing type')
    else:
        if np.isin(tracer,['pib','nav','fbb','fbp']):
            return  f'w{ptype.lower()}_cere_gaain.nii'
        if np.isin(tracer,['ftp','mk6240','pi2620']):
            return  f'w{ptype.lower()}_cereg_npdka.nii'


def pettype2ref(pet_tracer):
    '''
    What reference region shall we use for QC images?
    '''
    if pet_tracer in ['PI2620','MK6240','FTP',]:
        return 'inferior-cerebellum'
    elif pet_tracer in ABETA_WCEREB:
        return 'whole-cerebellum'
    elif pet_tracer in ABETA_CEREB_GM:
        return 'cerebellar_gm'
    else:
        raise ValueError('Invalid PET Tracer')


def TPi2TPs(TP_i):
    '''
    Switch TP integer to TP string
    0 becomes v1
    1 becomes v2... obviously.
    '''
    return f'v{TP_i+1}'
    # We used to use this notation BL/Scan2/Scan3/....ScanN 
    #if TP_i==0:
    #    return 'BL'
    #else:
    #    return f'Scan_{TP_i+1}'

def glob_re(pattern, path_in, return1=False):
    '''
    Glob with a regular expression!
    I use this to filter folders inside the data_path directory
    '''
    strings = glob.glob(path_in)
    F0=[]
    for s in strings:
        match = re.search(pattern, s)
        if match:
            if return1:
                F0.append(match.string)
            else:
                F0.append((match.string,match.group(1)))
    return F0

def empty_row(df,long_scan_info):
    empty_bool = pd.isnull(df.loc[long_scan_info,['Image Notes','QC Notes','Reviewer Initials']].values).all()
    return empty_bool


def move_tsv_header_to_first_line(filepath,QC_index_cols):
    '''
    Sometimes the Google spreadsheet is sorted without the top row being frozen.
    If this happens, the header could end up somewhere else. That's not fun.
    This code sorts the TSV files so that lines starting with "ID" are first
    '''
    print('Move TSV Header to first line...')
    df = pd.read_csv(filepath,low_memory=False,sep='\t',header=None,index_col=None)
    headers = df.loc[(df.iloc[:,:len(QC_index_cols)]  == QC_index_cols).all(axis=1)]
    values=df.loc[~(df.iloc[:,:len(QC_index_cols)]  == QC_index_cols).all(axis=1)]
    df = pd.concat([headers,values],axis=0,sort=True)
    df.to_csv(filepath,header=False,index=False,sep='\t',quoting=csv.QUOTE_ALL)
    print('Saved')

def aggregate_old_csvs(other_paths,aggregate_out_path):
    print('Aggregating old dataframes')
    if len(other_paths) == 0:
        return
    print('There are',len(other_paths),'old CSVs.')

    df_list=[]
    dlist=[]
    for p in other_paths:
        try:
            move_tsv_header_to_first_line(p,QC_index_cols)
            df = pd.read_csv(p,low_memory=False,dtype=str,sep='\t',header=0)
            #df = df.loc[~(df['ID'] == 'ID')]
            df = df.set_index(QC_index_cols)
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.replace('',np.nan)
            df = df.dropna(subset=['Reviewer Initials','Image Notes','QC Notes'],axis=0,how='all')

            df_list.append(df.copy())
            dlist.append(p)
            print('Aggregated',p)
        except:
            print('ERROR Error error - Cannot load other path',p)
    
    df_concat = pd.concat(df_list,ignore_index=False,sort=True)
    df_concat = df_concat[~df_concat.index.duplicated(keep='first')]

    df_concat.to_csv(aggregate_out_path,header=True, sep='\t', index=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
    if os.path.exists(aggregate_out_path):
        for p in dlist:
            pass
            #print('Removing',p)
            #os.remove(p)
    return




def Get_QC_CSV():
    today_date = Date.today()
    search_str = os.path.join(output_path,f'QC_{project_name}_{pet_tracer}-{ptype}_*.tsv')
    print(f'Searching for existing QC CSV: {search_str}')
    Old_QC_paths = glob.glob(search_str)
    Old_QC_paths.sort(key=lambda date: datetime.strptime(re.search(r'\d{2}-\d{2}-\d{4}', date).group(), "%m-%d-%Y"))
    dlist = []
    for p in Old_QC_paths:
        match = re.search(r'\d{2}-\d{2}-\d{4}', p)
        date = datetime.strptime(match.group(), '%m-%d-%Y').date()
        if (abs(today_date - date).days > 90) and ((len(Old_QC_paths) - len(dlist)) > 90):
            dlist.append(p)
    for p in dlist:
        print(f'(**Feature on suspend**) removing old file: {p}')
        #os.remove(p)
    Old_null_path = os.path.join(output_path,f'NULL_{project_name.upper()}_{pet_tracer.upper()}-{ptype.upper()}.tsv')
    #Old_null_paths.sort(key=lambda date: datetime.strptime(re.search(r'\d{2}-\d{2}-\d{4}', date).group(), "%m-%d-%Y"))
    dlist = []
    #for p in Old_null_paths:
    #    match = re.search(r'\d{2}-\d{2}-\d{4}', p)
    #    date = datetime.strptime(match.group(), '%m-%d-%Y').date()
    #    if (abs(today_date - date).days > 90) and ((len(Old_null_paths) - len(dlist)) > 90):
    #        dlist.append(p)
    #for p in dlist:
    #    print(f'removing old file: {p}')
    #    os.remove(p)

    # This list was sorted, last file is the newest:
    if len(Old_QC_paths) == 0:
        Old_QC_path = None
        other_paths=[]
    else:
        Old_QC_path = Old_QC_paths[-1]
        other_paths = Old_QC_paths[:-1]

   
    print('OLD QC PATH:',Old_QC_path)
    return Old_QC_path, other_paths

def Initalize_New_QC_CSV(Old_QC_path):
    '''
    Runs at start of QC code.
    Search for an existing QC CSV.
    Load data, set index, replace whitespace, replace empty with NaN, add missing columns if applicable

    Return: qc dataset, new qc path, new NULL path (same as the old one, no dates).
    '''
    new_qc_name = 'QC_{}_{}-{}_{}.tsv'.format(project_name.upper(),pet_tracer.upper(),ptype.upper(),Date.today().strftime("%m-%d-%Y"))
    new_qc_path = os.path.join(output_path,new_qc_name)
    NULL_out_path = os.path.join(output_path,'NULL_{}_{}-{}.tsv'.format(project_name.upper(),pet_tracer.upper(),ptype.upper()))
    aggregate_out_path = os.path.join(output_path,'CONCAT_{}_{}-{}.tsv'.format(project_name.upper(),pet_tracer.upper(),ptype.upper()))

    if Old_QC_path == None:
        df = pd.DataFrame(columns=QC_columns).set_index(QC_index_cols)
        df.to_csv(new_qc_path,header=True, sep='\t', index=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print('Loading existing QC Data file...')
    try:
        if Old_QC_path == None:
            Old_QC_path = new_qc_path

        move_tsv_header_to_first_line(Old_QC_path,QC_index_cols)
        df = pd.read_csv(Old_QC_path,low_memory=False,dtype=str,sep='\t',header=0)
        #df = df.loc[~(df['ID'] == 'ID')]
        df = df.set_index(QC_index_cols)

        idx = df.index
        if mrifree:
            df.index = df.index.set_levels([idx.levels[0].astype(str), idx.levels[1].astype(str), idx.levels[2].astype(str)])
        else:
            df.index = df.index.set_levels([idx.levels[0].astype(str), idx.levels[1].astype(str), idx.levels[2].astype(str), idx.levels[3].astype(str)])
        #df.dropna(axis=0, how='all', inplace=True)
    except:raise ValueError('Cannot read old QC CSV:',Old_QC_path)

    # Replace whitespace with NAN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace('',np.nan)
    df=df.sort_index()
    missing_columns = [i for i in QC_columns if i not in (list(df.columns.values) + list(df.index.names))]
    print('Missing columns',missing_columns)
    for insert_col in missing_columns:
        df[insert_col] = np.nan
    print('Initalized as null')
    df = df[QC_columns_noindex]
    return df,new_qc_path,NULL_out_path,aggregate_out_path

    

def create_QC_row(df,long_scan_info):
    '''
    If long_scan_info not in DF: Initalize a new row for a subject/PET combo. 
    If there exist mutliple rows for the same PET scan, nullify the rows with old MRIs.
    '''
    df = df.copy()
    if long_scan_info not in list(df.index.values):
        print('Missing subject in QC spreadsheet:',long_scan_info)
        df.loc[long_scan_info] = np.nan
    if (len(df.loc[long_scan_info[0], :,long_scan_info[2],:]) > 1) and (not mrifree):
        print('Found multiple MRI for single PET img. Will remove old MRI matches.')
        # below is a list of (subject,TP, PET_date, MRI_Date) values with all the same subject, TP, and PET_date:
        tp_mris=list(df.loc[long_scan_info[0],:,long_scan_info[2],:].index.values)
        tp_mris = [(long_scan_info[0],i[0],long_scan_info[2],i[1]) for i in tp_mris]
        tp_mris = [i for i in tp_mris if "__" not in i[1]] # We don't want to include analyses where the info was already nulled.
        print('TP MRIS:')
        print(tp_mris)
        ## This will sort nearest to the PET image date
        old_mris = [i for i in tp_mris if i != long_scan_info]
        for om in old_mris:
            # Null the TP index col
            # (subject, timepoint, PET date, MRI date)
            nulled_info = (om[0],'__'+om[1],om[2],om[3])
            i=1
            while nulled_info in list(df.index):
                print('Nulled info exists, adding 1',nulled_info)
                nulled_info = (om[0],'__'+om[1]+f'_{i}',om[2],om[3])
                i+=1
            print('Nulling / renaming',nulled_info)
            df.loc[nulled_info] = df.loc[om]
            df.drop(om,inplace=True)
            # Add Replaced MRI note:
            if pd.isnull(df.loc[nulled_info,['QC Notes','Image Notes']].values).all():
                df.drop(nulled_info,inplace=True)
            else:
                if pd.isnull(df.loc[nulled_info,'QC Notes']):
                    df.loc[nulled_info,'QC Notes']='Replaced MRI'
                else:
                    df.loc[nulled_info,'QC Notes']+=' - Replaced MRI'
    return df

def null_QC_row(df,long_scan_info, note='Reprocessed on XNAT'):
    '''
    Takes an index and nulls it by adding __ to the TP
    '''
    df = df.copy()
    if mrifree:
        nulled_subj = (long_scan_info[0],'__'+long_scan_info[1],long_scan_info[2])
    else:
        nulled_subj = (long_scan_info[0],'__'+long_scan_info[1],long_scan_info[2],long_scan_info[3])
    i=1
    while nulled_subj in list(df.index.values):
        print('Nulled info exists, adding 1',nulled_subj)
        i+=1
        if mrifree:
            nulled_subj = (long_scan_info[0],'__'+long_scan_info[1]+f'_{i}',long_scan_info[2])
        else:
            nulled_subj = (long_scan_info[0],'__'+long_scan_info[1]+f'_{i}',long_scan_info[2],long_scan_info[3])

    if empty_row(df,long_scan_info) == False:
        ''' If the row contains notes or a reviewer initial, put it into the NULL dataframe '''
        df.loc[nulled_subj] = df.loc[long_scan_info]
        if pd.isnull(df.loc[nulled_subj,'QC Notes']):
            df.loc[nulled_subj,'QC Notes']=note
        else:
            df.loc[nulled_subj,'QC Notes']+=f' - {note}'
    df.drop(long_scan_info,inplace=True)

    return df

def Update_QC_row(df,long_scan_info):
    '''
    When a QC image is recreated,  we need to update the QC row and null/archive the old rows as the image needs to be QC'd again.
    '''
    # If subject exists in df, null it.
    df = null_QC_row(df,long_scan_info)
    # Now we need to recreate the row with new scan info
    df = create_QC_row(df,long_scan_info)
    return df

def Add_MRIfree_Auto_QC_Flags(df,long_scan_info,PET_img):
    print('Checking / Updating automated flags.')
    df = df.copy()
    AUTOMATED_FLAGS = ['PET Cerebrum Overlap','PET Cerebellum Overlap']
    auto_df_values = df.loc[long_scan_info,AUTOMATED_FLAGS].values
    if pd.isnull(auto_df_values).any():
        print('Detected missing automated QC flags, running automated QC.')
        try:
            APARC = '/home/jagust/mboswell/Proj/xnat-proj/xnat-pipelines/PET_MRIFree/pipeline/PET_MRIFree/scripts/catalog/PET_MRIFree/scripts/lib/templates/NPDKA/NPDKA_rADNICN200_aparc+aseg_smoo1.50_normalized.nii'
            P_cerebrum,P_cerebellum = pet_bounding_box(APARC,PET_img)
            df.loc[long_scan_info,'PET Cerebellum Overlap'] = P_cerebellum
            df.loc[long_scan_info,'PET Cerebrum Overlap'] = P_cerebrum
        except Exception as e:
            print('Could not calculate automated PET bounding box')
            print(e)
    return df

def Add_Auto_QC_Flags(df,long_scan_info,MRI_img,baseline_MRI,PET_img,aparc_img):
    print('Checking / Updating automated flags.')
    df = df.copy()
    AUTOMATED_FLAGS = ['L/R Symmetry','PET Cerebrum Overlap','PET Cerebellum Overlap','Longitudinal MRI Alignment']
    auto_df_values = df.loc[long_scan_info,AUTOMATED_FLAGS].values
    print('Auto QC values in metadata:',auto_df_values)
    if pd.isnull(auto_df_values).any():
        print('Detected missing automated QC flags, running automated QC.')
        try:
            # Note: See automated_flags.py file for code
            LR_sym = LR_symmetry(aparc_img)
            df.loc[long_scan_info,'L/R Symmetry'] = LR_sym

            P_cerebrum,P_cerebellum = pet_bounding_box(aparc_img,PET_img)
            df.loc[long_scan_info,'PET Cerebellum Overlap'] = P_cerebellum
            df.loc[long_scan_info,'PET Cerebrum Overlap'] = P_cerebrum
            longi_correlation = image_correlation(src=MRI_img,ref=baseline_MRI)
            df.loc[long_scan_info,'Longitudinal MRI Alignment'] = longi_correlation
        except Exception as e:
            print('Could not calculate automated PET bounding box')
            print(e)
    return df

def Save_QC_df(QC_df,new_qc_path,NULL_out_path):
    # Save QC CSV
    QC_df = QC_df.sort_index()
    QC_df = QC_df.replace(r'^\s*$', np.nan, regex=True)
    QC_df = QC_df.replace('',np.nan)
    print('New review file --> ',new_qc_path)
    try:
        # Create subset of QC_df containing nulled rows.
        df_null = QC_df.loc[QC_df.index.get_level_values("TP").str.contains("__[v\d{1,3} | BL | Scan\d{1,3}]")]
        df_null = df_null.sort_index()
        print('Updating NULL df -',NULL_out_path)
        # To remove NULL from regular output file, uncomment below
        # To include NULL in regular output file, comment out below 
        for null_i in list(df_null.index.values):
            # If image notes and reviewer initials are null, we can just drop the QC row.
            if all(pd.isnull(QC_df.loc[null_i,['Image Notes','Reviewer Initials']])) == True:
                QC_df.drop(null_i,inplace=True)
                df_null.drop(null_i,inplace=True)
            else:
            #    # If it has been QC'd or there is a note, we want to keep it in the null DF.
                QC_df.drop(null_i,inplace=True)
        if os.path.exists(NULL_out_path):
            # If old NULL file exists, load it, append it to the new one, and drop rows where index and values are duplicates, then save.
            df_null_old = pd.read_csv(NULL_out_path,header=0,index_col=QC_index_cols,dtype=str,low_memory=False,sep='\t')
            df_null = pd.concat([df_null_old,df_null], axis=0,sort=True)
            df_null = df_null[~((df_null.index.duplicated(keep='first')) & (df_null.duplicated(keep='first')))]

        df_null.to_csv(NULL_out_path,header=True, sep='\t', index=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
    except Exception as e:print(e)
    print('Saving -',new_qc_path)
    QC_df = QC_df.replace(r'^\s*$', np.nan, regex=True)
    QC_df = QC_df.replace('',np.nan)
    QC_df.to_csv(new_qc_path,header=True, sep='\t', index=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
    QC_short_df = QC_df.copy()
    


def checkIsBusy(project,tracer):
    localBusySign = os.path.join(f'/home/jagust/petcore/qc/scripts/required/BusySign_{project}_{tracer}.txt')
    if os.path.exists(localBusySign):
        return True
    else:
        return False

def updateIsBusy(project,tracer,direction):
    localBusySign = os.path.join(f'/home/jagust/petcore/qc/scripts/required/BusySign_{project}_{tracer}.txt')
    if direction == 'up':
        if os.path.exists(localBusySign) == False:
            print('Creating -',localBusySign)
            with open(localBusySign, 'w') as out_:
                out_.write('QC code busy.')
    elif direction == 'down':
        if os.path.exists(localBusySign) == True:
            print('Deleting -',localBusySign)
            os.remove(localBusySign)
    else:
        print('Direction not defined as up or down')


def run_qc(project_name,pet_tracer,ptype,data_path,sub_match, output_path,remove_lost_data=False,align_QC_img=False, inorm=False,mrifree=False):
    ''' 
    project_name : ADNI/BACS
    pet_tracer    : tracer FTP/FBB/PIB/FBP 
    ptye        : processing type SUVR/DVR
    data_path   : path to squid project
    sub_match   : regex pattern for finding subjects
                  This can also be a specific subject
    output_path : Folder where QC images and meta will go.

    What is this doing:
        1.) Load / create new QC CSV.
        2.) Manage QC images, making new ones and deleting old ones.
        3.) 

    '''
    if not os.path.exists(output_path):  os.makedirs(output_path,exist_ok=True)

    # Get the path of an existing QC CSV or create one if it doesn't exist
    old_qc_csv_path,other_paths = Get_QC_CSV()
    print('Most recent QC CSV:', old_qc_csv_path)
    # Open the old dataframe and copy it 
    QC_df,new_qc_path,NULL_out_path,aggregate_out_path = Initalize_New_QC_CSV(old_qc_csv_path)
    QC_df_copy=QC_df.copy()

    # Grab all the subject data paths
    # This is specific to SQUID formatting
    # (?:BL|Scan\d{1,2})
    SUBJECTS_PATHS = glob_re(os.path.join(data_path , f'({sub_match})'), os.path.join(data_path,'*'))

    #SUBJECTS_PATHS = [i for i in SUBJECTS_PATHS if ('000-S-' not in i)]
    

    # This will track all images a QC image is made for.
    # If there exists an image not in this list at the end of the run, it will delete them.
    EXPECTED_IMG_LIST=[]
    EXPECTED_QC_ROWS=[]
    today_datetime = Date.today()
    
    # Print subjects found
    LEN_SUBJECT_PATHS = len(SUBJECTS_PATHS)
    print(f'New scan count: {LEN_SUBJECT_PATHS}')
    pet_filename = pettype2name(ptype.lower(),pet_tracer.lower(),mrifree) 
    reference_region = pettype2ref(pet_tracer)
    save_counter=1
    for i,(sub_path,sub) in enumerate(SUBJECTS_PATHS[:]):
        #if sub != '006-S-6674': continue
        #if not QC_df.equals(QC_df_copy):
        #    save_counter+=1
        #    if (save_counter%100 == 0):
        #        Save_QC_df(QC_df,new_qc_path,NULL_out_path)
        #save_counter+=1
        #if (save_counter%10 == 0):
        #    Save_QC_df(QC_df,new_qc_path,NULL_out_path)
        # For a subject, get all the PET folders matching a given tracer. 
        # Sort so they are in order of "timepoint".
        PET_folders = glob_re(os.path.join(sub_path,r'PET_(\d{4}-\d{2}-\d{2})_'+pet_tracer+r'(_|$).*'),os.path.join(sub_path,f'PET_*{pet_tracer}*'))
        PET_folders = [i for i in PET_folders if os.path.exists(os.path.join(i[0],'analysis'))]
        PET_folders.sort(key=lambda date: datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', date[1]).group(), "%Y-%m-%d"))
        if len(PET_folders) > 0:
            baseline_MRI = os.path.realpath(os.path.join(PET_folders[0][0],'analysis','rnu.nii'))
        for TP_i,(PET_folder,PET_date) in enumerate(PET_folders):
            # Convert 0-n with string BL/Scan2/.../ScanN-1
            TP_s = TPi2TPs(TP_i)
            # For PET scan matching the expected tracer, make a QC image
            PET_folder = os.path.join(PET_folder,'analysis')
            
            # We don't want the full path to this symlink, just the name of the MRI folder so we can get date from it
            mri_folder = os.path.realpath(os.path.join(PET_folder,'mri'))
            try:MRI_date = re.search(r'MRI_(\d{4}-\d{2}-\d{2})', os.path.basename(mri_folder)).group(1)
            except:MRI_date = 'MRI-DATE-FAILED'
            # Identifying info for row in QC spreadsheet
            if mrifree:long_scan_info = (sub,TP_s,PET_date)
            else:long_scan_info = (sub,TP_s,PET_date,MRI_date)
            short_scan_info = (sub,TP_s,PET_date)
            
            PET_img = os.path.join(PET_folder, pet_filename)
            MRI_img = os.path.realpath(os.path.join(PET_folder, 'rnu.nii'))
            APARC_img   = os.path.realpath(os.path.join(PET_folder, 'raparc+aseg.nii'))
            ICGM_img    = os.path.realpath(os.path.join(PET_folder, 'inf_cerebellar_mask.nii'))
            essential_imgs = [PET_img, MRI_img, APARC_img]

            # Add important masks via lines like beloww
            if reference_region == 'inferior-cerebellum':
                essential_imgs.append(ICGM_img)


            if mrifree:
                APARC_img = template_aparc 
                essential_imgs = [PET_img,APARC_img]    
            if not all([os.path.exists(i) for i in essential_imgs]):
                print ('Essential images:',essential_imgs)
                print(long_scan_info, 'Missing essential volumes')
                continue
            output_file_path = os.path.join(output_path, TP_s)
            output_file_name = f'{sub}_{TP_s}_{PET_date}_{pet_tracer}-{ptype}.png'
            img_output = os.path.join(output_file_path,output_file_name)
            
            # At the end, we compare all the existing QC images with the expected QC images and delete the extras 
            #   if the flag remove_lost_data == True
            #####
            #
            # Check if followup PET scan is less than 3 months old. If so, skip. 
            #
            ####
            PET_datetime = datetime.strptime(PET_date, "%Y-%m-%d").date()
            if TP_i != 0:
                d_delta = abs((PET_datetime - today_datetime).days)
                if d_delta <= 90:
                    print('PET is a followup that is less than 90 days old.')
                    print('Skipping')
                    continue
            
            EXPECTED_IMG_LIST.append(img_output)
            
            # Run create_QC_row, checks if long_scan_info is in the dataframe and checks for duplicate PET rows.
            QC_df = create_QC_row(QC_df,long_scan_info)
            EXPECTED_QC_ROWS.append(long_scan_info)

            if os.path.exists(img_output) == True:
                print('Found an img here already:')
                ''' 
                If there is a QC image by this name, lets check it for new PET / MRI filedates 
                If the modify date has not changed, there is no reason to check for changed data
                '''
                try:
                    QC_img_meta = Image.open(img_output).text
                    image_existed_before_run = True
                    if not mrifree and (np.isin(project_name.lower(), ['adni','pointer','bacs','ucsf'])):
                        # Old QC image code - To be simplified soon (tyler)
                        if ('PET_modify_date' in list(QC_img_meta.keys()) and 'MRI_modify_date' in list(QC_img_meta.keys())) == True:
                            # If PET and MRI mod date is not in img meta, needs new qc image.
                            pet_mod_time = time.ctime(os.path.getmtime(os.path.realpath(PET_img)))
                            mri_mod_time = time.ctime(os.path.getmtime(os.path.realpath(MRI_img)))
                            print('PET mod date:', QC_img_meta['PET_modify_date'])
                            print('Expecting:',pet_mod_time)
                            print('MRI mod date:',QC_img_meta['MRI_modify_date'])
                            print('Expecting:',mri_mod_time)
                            if (QC_img_meta['PET_modify_date'] == pet_mod_time) and (QC_img_meta['MRI_modify_date'] == mri_mod_time):
                                # Make sure the auto QC flags are filled in
                                QC_df = Add_Auto_QC_Flags(QC_df,long_scan_info,MRI_img,baseline_MRI,PET_img,APARC_img)
                                print('****************')
                                print('PET and MRI file modify date is a match. No need to check SUVRs and volumes.')
                                print('Continue\n******************')
                                continue
                            else:
                                print('Modify of PET or MRI has changed since last running Visual QC code.')
                        else:
                            print('MRI used for this analysis has changed. Reprocess QC image.')
                    else:
                        if 'image_modify_date' in list(QC_img_meta.keys()):
                            pet_mod_time = time.ctime(os.path.getmtime(os.path.realpath(PET_img)))
                            print('PET mod date:', QC_img_meta['image_modify_date'])
                            print('Expecting:',pet_mod_time)
                            if (pet_mod_time == QC_img_meta['image_modify_date']):
                                QC_df = Add_MRIfree_Auto_QC_Flags(QC_df,long_scan_info,PET_img)
                                print('****************')
                                print('PET modify date is a match. No need to check SUVRs')
                                print('Continue\n******************')
                                continue
                except Exception as e:
                    print(e)
                    print('Image metadata is corrupt and can not be read.')
                    print('Reprocessing QC image...')

                    

            else:
                ''' If it does not exist or has been recently modified, keep goin down the code...'''
                image_existed_before_run = False
                pass
            print(f'Creating {sub} {TP_s} QC image')
            reference = pd.DataFrame(columns=['folder','mask','value'])
            aparc_basename = os.path.basename(APARC_img)
            if reference_region == 'inferior-cerebellum':
                reference.loc[len(reference)] = [PET_folder, 'inf_cerebellar_mask.nii',1]
            #elif reference_region == 'inferior-cerebellum80':
            #    reference.loc[len(reference)] = [PET_folder, 'inf_cerebellar_mask80.nii',1]
            elif reference_region == 'whole-cerebellum':
                reference.loc[len(reference)] = [PET_folder, aparc_basename,46]
                reference.loc[len(reference)] = [PET_folder, aparc_basename,47]
                reference.loc[len(reference)] = [PET_folder, aparc_basename,7]
                reference.loc[len(reference)] = [PET_folder, aparc_basename,8]
            elif reference_region == 'cerebellar_gm':
                reference.loc[len(reference)] = [PET_folder, aparc_basename,8]
                reference.loc[len(reference)] = [PET_folder, aparc_basename,47]
            else:
                raise ValueError('Bad reference region!')
    
            display_df = pd.DataFrame(columns=['folder','mask','PET_mask','MRI_mask','LUT_name','erode'])
            ''' 
            Note: When using or making one of the composite regions with prefix QC_, you need to tell draw_func to look for the composite region in the LUT
            '''

            if pet_tracer in TAU_TRACERS:
                display_df.loc[len(display_df)] = [PET_folder, 'inf_cerebellar_mask.nii',0,1,'inf_cerebellar_mask',0]
                display_df.loc[len(display_df)] = [PET_folder, 'inf_cerebellar_mask.nii',1,0,'inf_cerebellar_mask_pet',2]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_Cortex',1]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_Tau_Regions',1]
                VMIN,VMAX = 0,2.5
            #elif pet_tracer == 'MK6240':
            #    display_df.loc[len(display_df)] = [PET_folder, 'inf_cerebellar_mask80.nii',0,1,'inf_cerebellar_mask',0]
            #    display_df.loc[len(display_df)] = [PET_folder, 'inf_cerebellar_mask.nii',0,1,'white',1]
            #    display_df.loc[len(display_df)] = [PET_folder, 'inf_cerebellar_mask80.nii',1,0,'inf_cerebellar_mask_pet',3]
            #    display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_Cortex',1] 
            #    display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_Tau_Regions',1]
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),1,0,'QC_Cortex_white',1]
            #    #display_df.loc[len(display_df)] = ['/home/tyler/tmp_imgs/45377680/whitemask70_nuspace_aparc.nii',1,0,'whitemask70_nuspace_aparc_pet',1]
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),0,1,'QC_B34',0]
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),1,0,'QC_B56_pet',1]
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),0,1,'ctx-rh-entorhinal',0] 
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),0,1,'ctx-lh-entorhinal',0] 
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),1,0,'ctx-rh-entorhinal_pet',1] 
            #    #display_df.loc[len(display_df)] = [os.path.join(sub_path,'mri','aparc+aseg.nii'),1,0,'ctx-lh-entorhinal_pet',1]
            #    VMIN,VMAX = 0,2.5
            elif pet_tracer in ABETA_WCEREB:
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_whole_cereb',0]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_whole_cereb_pet',2]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_Cortex',1]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_Cort_Sum_pet',1]
                VMIN,VMAX = 0,2.7
            elif pet_tracer in ABETA_CEREB_GM:
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_cereb_gm',0]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_cereb_gm_pet',2]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,0,1,'QC_Cortex',1]
                display_df.loc[len(display_df)] = [PET_folder, aparc_basename,1,0,'QC_Cort_Sum_pet',1]
                VMIN,VMAX = 0,2.7
            else:
                print('**************** BAD PET TYPE ********************')
                print('Aborting')
                continue
                #raise ValueError('PET TYPE NOT CODED FOR',pet_tracer)
    
            if os.path.exists(img_output):
                # This function will check if img_output exists, then verify correlation is 0.99, slope=1, intercept=0
                print('Running "check img" image to compare SUVR and VOLUMES')
                if (not mrifree) and np.isin(project_name.lower(), ['adni','pointer','bacs','ucsf']):
                    img_data_match = di.check_img(MRI_img, PET_img, img_output, \
                                display_df=display_df.copy(),reference=reference.copy(),\
                                alt_path_names=(MRI_img,PET_img),\
                                timepoint=(MRI_date,PET_date), \
                                VMIN = VMIN,VMAX= VMAX,inorm=inorm)
                else:
                    img_data_match = ucbqc.check_img(img_output,PET_img,APARC_img)
            else:
                img_data_match = False
            if img_data_match == True:
                print('Image reprocessed but unchanged. ')
                print('Updating QC image metadata.')
                if np.isin(project_name.lower(), ['adni','pointer','bacs','ucsf']) and (not mrifree):
                    di.update_metadata_dict(img_output, MRI_img, PET_img, display_df.copy(), reference.copy(), inorm)
                else:
                    ucbqc.update_metadata_dict(img_output,PET_img,APARC_img)

                print('Continuing to next image.')
                continue
            elif os.path.exists(img_output):
                print('QC image exists but is out of date:',long_scan_info)
                os.remove(img_output)
            else:
                print('QC image does not exist, continuing.')
            print('Drawing image')

            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)

            try:
                if align_QC_img == True:
                    random_code=''.join(random.choices(string.ascii_letters, k=6))
                    tmp_qc_folder = f'/home/jagust/petcore/NotBackedUp/tmpQCimg_{sub}_{TP_s}_{pet_tracer}-{ptype}_{random_code}'
                    while os.path.exists(tmp_qc_folder) == True:
                        # Safety check. If the tmp path exists, try a new random code. Don't overwrite an existing folder.
                        print('tmp folder exists, trying another.')
                        random_code=''.join(random.choices(string.ascii_letters, k=6))
                        tmp_qc_folder = f'/home/jagust/petcore/NotBackedUp/tmpQCimg_{sub}_{TP_s}_{pet_tracer}-{ptype}_{random_code}'
                    # If true, we are going to create a temp dir for processing and copy all needed files to it. This will fold aligned data.
                    os.makedirs(tmp_qc_folder,exist_ok=True)
                    tdisplay_df = display_df.copy()
                    treference = reference.copy()
                    shutil.copy(MRI_img, os.path.join(tmp_qc_folder,os.path.basename(MRI_img)), follow_symlinks=True)
                    shutil.copy(PET_img, os.path.join(tmp_qc_folder,os.path.basename(PET_img)), follow_symlinks=True)
                    for pc in list(np.unique((tdisplay_df['folder'] +'/'+ tdisplay_df['mask']).values)):
                        shutil.copy(pc, os.path.join(tmp_qc_folder,os.path.basename(pc)) , follow_symlinks=True)
                    for pc in list(np.unique((treference['folder'] +'/'+ treference['mask']).values)):
                        shutil.copy(pc, os.path.join(tmp_qc_folder,os.path.basename(pc)) , follow_symlinks=True)
                    if not os.path.exists(os.path.join(tmp_qc_folder,os.path.basename(APARC_img))):
                        shutil.copy(APARC_img, os.path.join(tmp_qc_folder,os.path.basename(APARC_img)) , follow_symlinks=True)
                    tdisplay_df['folder'] = tmp_qc_folder
                    treference['folder'] = tmp_qc_folder

                    tA=list(np.unique((tdisplay_df['folder'] +'/'+ tdisplay_df['mask']).values))
                    tB=['-n',]*len(tA)
                    tresult = [None]*(len(tA)+len(tB))
                    tresult[::2] = tB
                    tresult[1::2] = tA

                    tMRI_img = os.path.join(tmp_qc_folder,os.path.basename(MRI_img))
                    tPET_img = os.path.join(tmp_qc_folder,os.path.basename(PET_img))
                   
                    # Run alignment
                    print(' '.join(['/home/jagust/petcore/qc/scripts/required/nifti_align',\
                            '-i',os.path.join(tmp_qc_folder,os.path.basename(APARC_img)),\
                            '-l',tMRI_img,\
                            '-l',tPET_img]\
                            +tresult+[\
                            '--template','/home/jagust/petcore/qc/scripts/required/rniftialigntemplate_fsavg35_MNI152_aparc+aseg.nii']))
                    subprocess.run(['/home/jagust/petcore/qc/scripts/required/nifti_align',\
                            '-i',os.path.join(tmp_qc_folder,os.path.basename(APARC_img)),\
                            '-l',tMRI_img,\
                            '-l',tPET_img]\
                            +tresult+[\
                            '--template','/home/jagust/petcore/qc/scripts/required/rniftialigntemplate_fsavg35_MNI152_aparc+aseg.nii'],\
                            stdout=open(os.devnull, 'wb'))
                    
                    # change filenames for aligned versions
                    tdisplay_df['mask'] = tdisplay_df['mask'].apply(lambda x: f"a_{x}")
                    treference['mask'] = treference['mask'].apply(lambda x: f"a_{x}")
                    tMRI_img = os.path.join(tmp_qc_folder,'a_'+os.path.basename(MRI_img))
                    tPET_img = os.path.join(tmp_qc_folder,'a_'+os.path.basename(PET_img))
                    tAPARC_img = os.path.join(tmp_qc_folder,'a_'+os.path.basename(APARC_img))
                    tICGM_img = os.path.join(tmp_qc_folder,'a_'+os.path.basename(ICGM_img))

                    # Run draw image code with the temporary aligned filenames
                    #reference = treference
                    #display_df = tdisplay_df
                    if np.isin(project_name.lower(), ['adni','pointer','bacs','ucsf'] ):
                        print('draw_image',tMRI_img,tPET_img,img_output,'display_df','reference',(MRI_date,PET_date),VMIN,VMAX,False)
                        di.draw_image(tMRI_img, tPET_img, img_output, \
                                display_df=tdisplay_df,reference=treference,\
                                alt_path_names=(MRI_img,PET_img,display_df,reference),\
                                timepoint=(MRI_date,PET_date), \
                                VMIN = VMIN,VMAX= VMAX,inorm=inorm)
                    else:
                        print('UCB Drawing Function -',img_output)
                        print([tMRI_img,tPET_img,tAPARC_img,tICGM_img])
                        print(pet_tracer)
                        print([MRI_img,PET_img])
                        print([MRI_date,PET_date])
                        ucbqc.draw_mridependent(img_output,\
                                                [tMRI_img,tAPARC_img,tPET_img,tICGM_img],\
                                                pet_tracer.upper(),\
                                                xnat_paths=[MRI_img,APARC_img,PET_img,ICGM_img],\
                                                image_dates=[MRI_date,PET_date])
                        print('*- draw_mridependent finished')

                    if os.path.exists(tmp_qc_folder):
                        print('Cleaning',tmp_qc_folder)
                        subprocess.run(['rm','-rf',tmp_qc_folder])
                else:
                    if mrifree:
                        print('Drawing MRI-free')
                        ucbqc.draw_mrifree(img_output,PET_img,pet_tracer.upper(),display_path_name=PET_img,image_date=PET_date)
                    else:
                        print('Drawing MRI-dependent data')
                        if np.isin(project_name.lower(), ['adni','pointer','bacs','ucsf']): 
                            di.draw_image(MRI_img, PET_img, img_output, \
                                    display_df=display_df,reference=reference,\
                                    alt_path_names=(MRI_img,PET_img,display_df,reference),\
                                    timepoint=(MRI_date,PET_date), \
                                    VMIN = VMIN,VMAX= VMAX,inorm=inorm)
                        else:
                            ucbqc.draw_mridependent(img_output,
                                                    [MRI_img,APARC_img,PET_img,ICGM_img],
                                                    pet_tracer.upper(),
                                                    xnat_paths=[MRI_img,APARC_img,PET_img,ICGM_img],
                                                    image_dates=[MRI_date,PET_date])
                if os.path.exists(img_output):
                    print('Drawing QC image was successful')
                    # Check if info was in the QC DF before running:
                    if long_scan_info in list(QC_df_copy.index.values):
                        # The subject had a QC row before this was ran and we just updated the images so we need to also update the row.
                        QC_df = Update_QC_row(QC_df,long_scan_info)
                        
                    if mrifree:
                        QC_df = Add_MRIfree_Auto_QC_Flags(QC_df,long_scan_info,PET_img)
                    else:
                        QC_df = Add_Auto_QC_Flags(QC_df,long_scan_info,MRI_img,baseline_MRI,PET_img,APARC_img)
                    
                else:
                    print('FAILED')
            except Exception as e: 
                print(e)
                print(f'Error drawing img: {sub_path}')
                continue
    
    # For every row in the old df that is not on squid, null it. 
    # This will only include PET scans which were removed from squid. It will not include reprocessed stuff.
    
    #not_on_squid = [i for i in QC_df.index.to_list() if i not in EXPECTED_QC_ROWS and '__' not in i[1]]
    #for removed_row in not_on_squid:
    #    QC_df = null_QC_row(QC_df,removed_row, note = 'Removed from XNAT')
        
    ALL_QC_IMGS = glob.glob(os.path.join(output_path,'**/*.png'),recursive=True)
    ORPHANED_IMGS = list(set(ALL_QC_IMGS) - set(EXPECTED_IMG_LIST))
    #print(EXPECTED_IMG_LIST)
    #print(ALL_QC_IMGS)
    print('Orphaned images:')
    for oi in ORPHANED_IMGS:
        print(oi)
    if remove_lost_data: 
        print(f'Cleaning {len(ORPHANED_IMGS)} orphaned images')
        for oip in ORPHANED_IMGS:
            print(oip)
            os.remove(oip)

        print('Done')
    else:
        print(f'There are {len(ORPHANED_IMGS)} orphaned QC images. There is a flag to delete these if you wish.')
    Save_QC_df(QC_df,new_qc_path,NULL_out_path)

    print('Old subjects found: {}'.format(len(QC_df.groupby(level=0))))
    print('Old individual scans: {}'.format(len(QC_df.index)))

    if not mrifree:
        print('Replaced MRI-PET pairs: {}'.format(len(QC_df.groupby(level=[0,1,2,3]))-len(QC_df.groupby(level=[0,1,2]))))

    print('New subjects found: {}'.format(len(QC_df_copy.groupby(level=0))-len(QC_df.groupby(level=0))))
    #print('New individual scans: {}'.format(len(QC_IMG_CSV.index)-len(df.index)))
    print('New individual scans: {}'.format('NOT IMPLEMENTED'))

    #print('Cleaning old paths')
    #aggregate_old_csvs(other_paths,aggregate_out_path)

    print('Complete.')



if __name__ == "__main__":
    main()
