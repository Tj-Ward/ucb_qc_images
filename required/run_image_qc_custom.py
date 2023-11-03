#!/usr/bin/env python

####
#
# Update March 2nd, 2022
# Redesigned to function on Squid for any project
# -tjw
#
#
# reference is the table we use for defining the masks we want to use for intensity normalization for the PET image
# row = [path , index]
# 
# This simple dataframe contains a path and index for each ROI. 
# The ROIs are joined in draw_func and a mean of the PET is used
# to intensity normalize the image
#
# This can also be left blank if inorm=False (see below) but currently it needs to be initalized and passed to draw_func
#
# In this example, I use an eroded inferior cerebellar mask if MK or whole cerebellum from the aparc+aseg if not MK
#
####


####
#
# display_df is the table we use for defining the masks we want to show on the QC image
# Just like reference, display_df uses rows to define the path to a mask file
# Additional columns help us define the ROIs and the way it is displayed
#
# row = [path , MRI overlay, PET overlay, color code, outline/filled]
#    path: path to the mask file
#    MRI overlay: True/False - Do you want this on the MRI?
#    PET overlay: True/False - Do you want this on the PET?
#    ROI:
#    outline/filled:
#       Filled = 0
#       Border = 1
#       Outline 1 voxel = 2
#       Outline 2 voxels = 3
#
# I have many pre-defined ROIs in draw_func which help keep the display_df smaller
# Below for MK, I point to the inferior cereb mask and the aparc+aseg
# The first two rows only show on the PET portion of the QC image
#   They call the LUT names 'inf_cerebellar_mask' and 'white'
#   The first row uses display type 0, the second uses display type 1
#   Display type 0 = Filled in
#   Display type 1 = Border / hollowed out mask
#   Display type 2 or 3 = Outline / Everything inside is the mask, the line itself is not in the mask
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
from datetime import datetime
import dateutil.parser as dparser
from datetime import date as Date

from PIL import Image,ImageOps,ImageFont,ImageDraw
from PIL.PngImagePlugin import PngInfo

# Draw image is my image QC generation function
sys.path.append('/home/jagust/petcore/qc/scripts')
import draw_func as di
import pathlib

from automated_flags import *

parser = argparse.ArgumentParser()
parser.add_argument("--pet",type=str, help="Dir of coregistered PET nifti img", required=True)
parser.add_argument("--mri",type=str, help="Dir of MRI nifti img", required=True)
parser.add_argument("--aparc",type=str, help="Dir of coregistered aparc+aseg nifti img", required=False)
parser.add_argument("--fullaparc", action='store_true', help="Display the entire aparc+aseg on MRI", default=False, required=False)
parser.add_argument("--masks",type=str,nargs='+', help="Directories of a masks. List as many as you have under the one flag.", required=False)
parser.add_argument("-o", "--outpath",type=str, help="QC Image Folder (Ex: /home/jagust/adni/adni_qc/visual_qc)", required=True)
parser.add_argument('--niftialign', action='store_true',help="Use nifti_align to straighted images? Default no.",default=False)
parser.add_argument("--scale",nargs=2,metavar=('MIN','MAX'), help="Min and Max of PET display range", required=True)
parser.add_argument('--inorm', action='store_true',help="Intensity normalize by whole cereb",default=False)

args = parser.parse_args()

#convert_names = {'AV1451':'FTP',
#                'AV45':'FBP'}

PET_TYPES=['FTP','MK6240','PI2620','FBB','FBP','PIB']
ABETA_WCEREB = ['FBB','FBP']
ABETA_CEREB_GM = ['PIB',]
TAU_TRACERS = ['MK6240','FTP','PI2620']


align_QC_img = args.niftialign
inorm=args.inorm
reference_region = 'whole-cerebellum' # 'cerebellar_gm'
VMIN,VMAX = args.scale
VMIN = float(VMIN)
VMAX = float(VMAX)

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
    print('PET file:', args.pet)
    print('QC image output:', args.outpath)
    print('i-norm images:',args.inorm)
    print('Nifti align:',args.niftialign)
    
    run_qc(pet_path=args.pet, mri_path=args.mri,aparc_path=args.aparc, mask_paths=args.masks, img_output=args.outpath,align_QC_img=align_QC_img, inorm=inorm)

def pettype2name(ptype,tracer):
    '''
    What is the name of the PET file?
    '''
    if tracer == 'pib':
        return f'{ptype.lower()}_cereg.nii'
    if tracer == 'ftp':
        return f'{ptype.lower()}_cereg.nii'
    if tracer == 'fbb':
        return f'{ptype.lower()}_cere.nii'
    if tracer == 'fbp':
        return f'{ptype.lower()}_cere.nii'

    else:
        raise ValueError('Error with PET processing type')


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


def run_qc(pet_path, mri_path, aparc_path, mask_paths, img_output, align_QC_img, inorm):
    #if os.path.exists(out_path):
    #    print('#################### ERROR ########################')
    #    print('Output already exists! Choose a different name.')
    #    print('Exiting without processing')
    #    return
    if not os.path.exists(os.path.dirname(img_output)): 
        print('#################### ERROR ########################')
        print('Output folder does not exist!',os.path.dirname())
        print('Exiting without processing')
        return
    # Get the path of an existing QC CSV or create one if it doesn't exist

    MRI_img = os.path.realpath(mri_path)
    PET_img = os.path.realpath(pet_path)
    APARC = os.path.realpath(aparc_path)
    if mask_paths == None:
        MASKs = []
    else:
        MASKs = [os.path.realpath(i) for i in mask_paths]
    
    try:MRI_date = dparser.parse(mri_path,fuzzy=True).strftime("%m-%d-%Y")
    except:MRI_date = 'MRI-DATE-FAILED'
    try:PET_date = dparser.parse(pet_path,fuzzy=True).strftime("%m-%d-%Y")
    except:PET_date = 'PET-DATE-FAILED'
    
    essential_imgs = [PET_img, MRI_img]
    print('Essential images:',essential_imgs)

    if not all([os.path.exists(i) for i in essential_imgs]):
        print('Missing essential volumes')
        raise ValueError('Missing essential volumnes')

    if os.path.exists(img_output) == True:
        print('Found an img here already:')
        ''' If there is a QC image by this name, lets check it for new PET / MRI filedates '''
        QC_img_meta = Image.open(img_output).text
        if ('PET_modify_date' in list(QC_img_meta.keys()) and 'MRI_modify_date' in list(QC_img_meta.keys())) == True:
            # If PET and MRI mod date is not in img meta, needs new qc image.
            pet_mod_time = time.ctime(os.path.getmtime(os.path.realpath(PET_img)))
            mri_mod_time = time.ctime(os.path.getmtime(os.path.realpath(MRI_img)))
            print('PET mod date:', QC_img_meta['PET_modify_date'])
            print('Expecting:',pet_mod_time)
            print('MRI mod date:',QC_img_meta['MRI_modify_date'])
            print('Expecting:',mri_mod_time)
            if (QC_img_meta['PET_modify_date'] == pet_mod_time) and (QC_img_meta['MRI_modify_date'] == mri_mod_time):
                print('****************')
                print('PET and MRI file modify date is a match. No need to check SUVRs and volumes.')
                return
            else:
                print('Modify of PET or MRI has changed since last running Visual QC code.')
        else:
            print('Modify date of PET or MRI is not in the QC image metadata. Rerunning QC image. ')

    else:
        ''' If it does not exist or has been recently modified, keep goin down the code...'''
        pass
    print(f'Creating QC image')
    reference = pd.DataFrame(columns=['folder','mask','value'])
    aparc_basename = os.path.basename(APARC)
    aparc_folder = os.path.dirname(APARC)
    if reference_region == 'whole-cerebellum':
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,46]
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,47]
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,7]
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,8]
    elif reference_region == 'cerebellar_gm':
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,8]
        reference.loc[len(reference)] = [aparc_folder, aparc_basename,47]
    else:
        raise ValueError('Bad reference region!')
    
    display_df = pd.DataFrame(columns=['folder','mask','PET_mask','MRI_mask','LUT_name','erode'])
    ''' 
    Note: When using or making one of the composite regions with prefix QC_, you need to tell draw_func to look for the composite region in the LUT
    '''
    # MRI row mask:
    if args.fullaparc:
        display_df.loc[len(display_df)] = [aparc_folder, aparc_basename,0,1,'QC_fullaparc',0]
    else:
        display_df.loc[len(display_df)] = [aparc_folder, aparc_basename,0,1,'QC_Cortex',1]
        display_df.loc[len(display_df)] = [aparc_folder, 'avg_inf_cerebellar_mask.nii',0,1,'inf_cerebellar_mask',1]
        display_df.loc[len(display_df)] = [aparc_folder, aparc_basename,1,0,'QC_Tau_Regions',1]
        #display_df.loc[len(display_df)] = [aparc_folder, aparc_basename,1,0,'QC_generic',1]
        for mask in MASKs:
            mask_folder=os.path.dirname(mask)
            mask_basename = os.path.basename(mask)
            display_df.loc[len(display_df)] = [mask_folder, mask_basename,0,1,'QC_generic_red',1]
            display_df.loc[len(display_df)] = [mask_folder, mask_basename,0,1,'QC_generic',2]
    
    print('Running "check img" image to compare SUVR and VOLUMES')

    # This function will check if img_output exists, then verify correlation is 0.99, slope=1, intercept=0
    img_data_match = di.check_img(MRI_img, PET_img, img_output, \
                display_df=display_df.copy(),reference=reference.copy(),\
                alt_path_names=(MRI_img,PET_img),\
                timepoint=(MRI_date,PET_date), \
                VMIN = VMIN,VMAX= VMAX,inorm=inorm)
    if img_data_match == True:
        print('Image unchanged. ')
        print('Updating QC image metadata.')
        di.update_metadata_dict(img_output, MRI_img, PET_img, display_df.copy(), reference.copy(), inorm)
        #di.update_metadata_dict(img_output, meta_)
        return
    elif os.path.exists(img_output):
        print('QC iamge exists but is out of date')
        os.remove(img_output)
    print('Image changed')
    print('Drawing image')

    try:
        if align_QC_img == True:
            homedir = os.path.expanduser('~')
            notbackedup = os.path.join(homedir,'NotBackedUp')
            if not os.path.exists(notbackedup):
                os.makedirs(notbackedup)
            random_code=''.join(random.choices(string.ascii_letters, k=6))
            temp_pet_name = os.path.basename(PET_img).replace('.','_').replace(' ','_')
            tmp_qc_folder = os.path.join(notbackedup,f'tmpQCimg_{temp_pet_name}_{random_code}')
            while os.path.exists(tmp_qc_folder) == True:
                # Safety check. If the tmp path exists, try a new random code. Don't overwrite an existing folder.
                print('tmp folder exists, trying another.')
                random_code=''.join(random.choices(string.ascii_letters, k=6))
                tmp_qc_folder = os.path.join(notbackedup,f'tmpQCimg_{temp_pet_name}_{random_code}')
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
            if not os.path.exists(os.path.join(tmp_qc_folder,os.path.basename(APARC))):
                shutil.copy(APARC, os.path.join(tmp_qc_folder,os.path.basename(APARC)) , follow_symlinks=True)
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
                    '-i',os.path.join(tmp_qc_folder,os.path.basename(APARC)),\
                    '-l',tMRI_img,\
                    '-l',tPET_img]\
                    +tresult+[\
                    '--template','/home/jagust/petcore/qc/scripts/required/rniftialigntemplate_fsavg35_MNI152_aparc+aseg.nii']))
            subprocess.run(['/home/jagust/petcore/qc/scripts/required/nifti_align',\
                    '-i',os.path.join(tmp_qc_folder,os.path.basename(APARC)),\
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

            # Run draw image code with the temporary aligned filenames
            print('draw_image',tMRI_img,tPET_img,img_output,'display_df','reference',(MRI_date,PET_date),VMIN,VMAX,False)
            #reference = treference
            #display_df = tdisplay_df
            print(display_df)
            print(reference)
            di.draw_image(tMRI_img, tPET_img, img_output, \
                        display_df=tdisplay_df,reference=treference,\
                        alt_path_names=(MRI_img,PET_img,display_df,reference),\
                        timepoint=(MRI_date,PET_date), \
                        VMIN = VMIN,VMAX= VMAX,inorm=inorm,cols=16)
            if os.path.exists(tmp_qc_folder):
                print('Cleaning',tmp_qc_folder)
                subprocess.run(['rm','-rf',tmp_qc_folder])
        else:
            print('draw_image',MRI_img,PET_img,img_output,'display_df','reference',(MRI_date,PET_date),VMIN,VMAX,False)
            di.draw_image(MRI_img, PET_img, img_output, \
                        display_df=display_df,reference=reference,\
                        alt_path_names=(MRI_img,PET_img,display_df,reference),\
                        timepoint=(MRI_date,PET_date), \
                        VMIN = VMIN,VMAX= VMAX,inorm=inorm,cols=16)
        if os.path.exists(img_output):
            print('Drawing QC image was successful')
        else:
            print('FAILED')
    except Exception as e: 
        print(e)
        return
    
    print('Complete.')



if __name__ == "__main__":
    main()
