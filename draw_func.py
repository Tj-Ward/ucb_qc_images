#!/usr/bin/env python
import numpy as np
import nibabel as nib
import os,re,sys,glob
import time
import pathlib
import subprocess

from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.colors as Colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import r2_score

from PIL import Image,ImageOps,ImageFont,ImageDraw
from PIL.PngImagePlugin import PngInfo
import cv2


# This is a color LUT for all the ROIs you want to use. 
# I started with the Freesurfer LUT for FSLeyes and added from there
Color_LUT= pd.read_csv('/home/jagust/petcore/qc/scripts/required/LUT_all.txt')

# Path to a font on your computer:
font_path = '/usr/share/fonts/gnu-free/FreeSans.ttf'

# This is just a list of cerebral GM ROIs I want to display
QC_image_cortical_rois = '/home/jagust/petcore/qc/scripts/required/QC_Image_Cortex_LUT.txt'

# Colormap for PET images:
NIH_colors = np.genfromtxt('/home/jagust/petcore/qc/scripts/required/NIH.csv', delimiter=',',dtype=float)
# Add a gradient of alpha to the highest voxel values:
NIH_colors[0:2, -1] = 0
NIH_colors[2:20, -1] = np.sqrt(np.linspace(0,1,18))
NIH = ListedColormap(NIH_colors)

BRAAK1 = [1006,2006]
BRAAK2 = [17,53]
BRAAK3 = [1016,1007,1013,18,2016,2007,2013,54]
BRAAK4 = [1015,1002,1026,1023,1010,1035,1009,1033,2015,2002,2026,2023,2010,2035,2009,2033]
BRAAK5 = [1028,1012,1014,1032,1003,1027,1018,1019,1020,1011,1031,1008,1030,1029,1025,1001,1034,2028,2012,2014,2032,2003,2027,2018,2019,2020,2011,2031,2008,2030,2029,2025,2001,2034]
BRAAK6 = [1021,1022,1005,1024,1017,2021,2022,2005,2024,2017]
META_TEMPORAL_CODES = [1006,2006,18,54,1007,2007,1009,2009,1015,2015]

QC_Tau_Regions = META_TEMPORAL_CODES
B34_codes = [1016,1007,1013,18,2016,2007,2013,54,1015,1002,1026,1023,1010,1035,1009,1033,2015,2002,2026,2023,2010,2035,2009,2033]
#QC_META_pet

B56_codes = [1028,1012,1014,1032,1003,1027,1018,1019,1020,1011,1031,1008,1030,1029,1025,1001,1034, \
                 2028,2012,2014,2032,2003,2027,2018,2019,2020,2011,2031,2008,2030,2029,2025,2001,2034, \
                 1021,1022,1005,1024,1017,2021,2022,2005,2024,2017]
Cort_Sum_codes = [1009,2009,1015,1030,2015,2030,1003,1012,1014,1018,1019,1020,1027,1028,1032,2003,2012,2014,\
                 2018,2019,2020,2027,2028,2032,1008,1025,1029,1031,2008,2025,2029,2031, \
                1015,1030,2015,2030,1002,1010,1023,1026,2002,2010,2023,2026]
QC_cortex_codes = list(pd.read_csv(QC_image_cortical_rois)['Index'].astype(int))
QC_whole_cereb_codes = [46,47,8,7]
QC_cereb_gm_codes = [47,8]
QC_generic_codes = QC_cortex_codes + QC_whole_cereb_codes

from io import StringIO
def read_csv(filename, comment='#', sep=','):
    lines = "".join([line for line in open(filename)
                     if not line.startswith(comment)])
    return pd.read_csv(StringIO(lines), sep=sep)

def unique_arr(array, orderby='first'):
    array = np.asarray(array)
    order = array.argsort(kind='mergesort')
    array = array[order]
    diff = array[1:] != array[:-1]
    if orderby == 'first':
        diff = np.concatenate([[True], diff])
    elif orderby == 'last':
        diff = np.concatenate([diff, [True]])
    else:
        raise ValueError
    uniq = array[diff]
    index = order[diff]
    return uniq[index.argsort()]

def prep_img(St_St, I='',MRI=False,PET=False,MASKs=False,erode_vals=False,X_size=256):
    '''
    2D slices.
    '''
    startx = St_St[0] -12
    stopx = St_St[1]+12
    starty = St_St[2]-12
    stopy = St_St[3]+12

    # Convert them both to PIL images
    MRI = MRI[startx:stopx,starty:stopy,:]
    MRI = np.flip(MRI,0)
    MRI = np.rot90(MRI,1)
    MRI = Image.fromarray(MRI)
    #MRI = ImageOps.scale(MRI, (X_size/MRI.size[0]), resample=Image.BILINEAR)

    if not isinstance(PET,bool):
        PET = PET[startx:stopx,starty:stopy,:]
        #PET[PET==0] = np.nan
        #PET = no_alpha_NIH(PET, bytes=True)
        PET = np.flip(PET,0)
        PET = np.rot90(PET,1)
        PET = Image.fromarray(PET)
        #PET = ImageOps.scale(PET, (X_size/PET.size[0]), resample=Image.BILINEAR)
        MRI.alpha_composite(PET, (0, 0))

    
    if not isinstance(MASKs,bool):
        for i,MASK in enumerate(MASKs):

            #print('MASK -',MASK.shape)
            #test_img = Image.fromarray(MASK.astype('uint8'), 'RGBA')
            ##test_img = ImageOps.scale(test_img, (X_size/test_img.size[0]), resample=Image.BILINEAR)
            #tsl = 0
            #tmp = f'/home/jagust/tjward/NotBackedUp/mask_{tsl}.png'
            #while os.path.exists(tmp):
            #    tsl+=1
            #    tmp = f'/home/jagust/tjward/NotBackedUp/mask_{tsl}.png'
            #test_img.save(tmp)

            erode_val = int(float(erode_vals[i]))
            MASK = MASK[startx:stopx,starty:stopy,:]
            MASK = np.flip(MASK,0)
            MASK = np.rot90(MASK,1)

            if erode_val != 0:
                MASK[:,:,3] = (MASK[:,:,3] > 0).astype(int) * 255
                vol = np.copy(MASK)
                vol = np.sum(vol[:,:,:],axis=2) # 2d mask
                vol = (vol>0).astype(int)
                #vol = (vol / np.nanmax(vol)) * 255
                #tmp_img = cm.gray(vol, bytes=True)
                omask = np.zeros(vol.shape + (4,))

                #cnts = cv2.findContours(vol.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                np.save('/home/jagust/tjward/NotBackedUp/vol.npy', vol.copy())
                # cv2.RETR_TREE,
                cnts = cv2.findContours(vol.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(omask, [c], -1, (255,255,255,255), thickness=erode_val)
                #omask[null_mask] = np.nan
                tmp_img = np.asarray(omask)
                #tmp_img = np.sum(tmp_img[:,:,:3],axis=2)
                MASK[tmp_img == 0] = 0
            mask_img = Image.fromarray(MASK.astype('uint8'), 'RGBA')
            #mask_img = ImageOps.scale(mask_img, (X_size/mask_img.size[1]), resample=Image.NEAREST)
            MRI.alpha_composite(mask_img, (0, 0))



       #     tmp_img = cm.gray(vol, bytes=True)
       #     omask = np.zeros(vol.shape + (4,))
       #     cnts = cv2.findContours(vol.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
       #     for c in cnts:
       #         cv2.drawContours(omask, [c], -1, colors, thickness=1)
       #     tmp_img = np.asarray(omask)
       #     MASK[:,:,:] = tmp_img

            #MASK[:,:,3] = MASK[:,:,3] * .6 # Add alpha to segmentation

    MRI = ImageOps.scale(MRI, (X_size/MRI.size[0]), resample=Image.LANCZOS)

    fnt = ImageFont.truetype(font_path, int(X_size//10))
    d = ImageDraw.Draw(MRI)
    d.text((0,0), str(I), font=fnt, fill=(255,255,255,255))

    return np.asarray(MRI)



def loadReqData(nu_path, pet_path, display_df:pd.DataFrame,reference:pd.DataFrame,inorm=False):
    print('Loading required data')
    check_paths_list = [nu_path,pet_path]

    for p in check_paths_list:
        if not os.path.exists(p):
            raise ValueError(f'*Missing file for image QC: {p}')

    # Load and normalize the mri to 0-255
    NU =  np.asarray(nib.load(nu_path).get_fdata()).astype(float)
    #NU = ((NU / (np.nanmean(NU) + (4 * np.nanstd(NU)))) * 235).astype(int)
    NU[NU > 200] = 200
    NU[NU < 0] = 0
    NU = NU / 200
    #NU = NU*255
    NU_colored = cm.gray(NU, bytes=True)
    interested_voxels = np.zeros(shape=NU.shape,dtype=bool)
    display_numpy_images = []#np.zeros(np.shape(NU_colored))
    display_numpy_images_pet = []#np.zeros(np.shape(NU_colored))
    erode_values = []
    erode_values_pet = []

    display_df['fullmask'] = (display_df['folder']+'/'+  display_df['mask'])
    reference['fullmask'] =  (reference['folder'] +'/'+  reference['mask'])
    print(reference['fullmask'])
    print(display_df['fullmask'])
    print('***************************')
    print('***************************')

    for mask_path in unique_arr(display_df['fullmask']):
        # for each unique mask file, open it and create the necessary PIL images.
        print('loading mask:',mask_path)
        mask_dat = np.round(np.asarray(nib.load(mask_path).get_fdata()),0)
        #color_lut_row = Color_LUT.loc[np.isin(Color_LUT['Name'] , display_df[display_df['mask'] == mask_path])]
        #color_lut_index = list(color_lut_row['Index'].astype(int))
        fullaparc=False

        for index, row in display_df[display_df['fullmask'] == mask_path].iterrows():
            show_pet = False
            show_mri = False
            if int(row['PET_mask']) == True:
                show_pet = True
            if int(row['MRI_mask']) == True:
                show_mri = True
            # Iterate over the LUT names of the ROIs
            roi_name = str(row['LUT_name'])
            #if roi_name == 'QC_FS_LUT':
            if 'QC_fullaparc' in roi_name:
                mask_subset = np.copy(mask_dat)
                fullaparc=True
            elif 'QC_Cortex' in roi_name:
                color_lut_index = QC_cortex_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_generic' in roi_name:
                color_lut_index = QC_generic_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_Tau_Regions' in roi_name:
                color_lut_index = QC_Tau_Regions
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_B34' in roi_name:
                color_lut_index = B34_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_B56' in roi_name:
                color_lut_index = B56_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_whole_cereb' in roi_name:
                color_lut_index = QC_whole_cereb_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_cereb_gm' in roi_name:
                color_lut_index = QC_cereb_gm_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            elif 'QC_Cort_Sum' in roi_name:
                color_lut_index = Cort_Sum_codes
                mask_subset = np.isin(mask_dat,color_lut_index)
            else:
                color_lut_row = Color_LUT.loc[Color_LUT['Name'] == roi_name]
                color_lut_index = list(color_lut_row['Index'].astype(int))
                if color_lut_index not in mask_dat:
                    continue
                if len(color_lut_index) > 1:
                    print('Warning (color_lut_index greater than 1):')
                    print('     There are multiple ROIs with the same name in the Color LUT')
                mask_subset = np.where(np.isin(mask_dat,color_lut_index) , mask_dat , 0)
            mask_subset = mask_subset.astype(int)
            #  if row['erode'] == 1:
            #      for ind in np.unique(mask_subset):
            #          if ind == 0:
            #              continue
            #          bin_ind = mask_subset == ind
            #          # Dilation method:
            #          bin_ind_di = binary_dilation(bin_ind, iterations=1)
            #          mask_subset[bin_ind_di] = ind
            #          mask_subset[bin_ind] = 0
            #  elif row['erode'] == 2:
            #      for ind in np.unique(mask_subset):
            #          if ind == 0:
            #              continue
            #          bin_ind = mask_subset == ind
            #          # Erosion method
            #          bin_ind_er = binary_erosion(bin_ind, iterations=1)
            #          mask_subset[bin_ind_er] = 0
            #  if row['erode'] == 3:
            #      for ind in np.unique(mask_subset):
            #          if ind == 0:
            #              continue
            #          bin_ind = mask_subset == ind
            #          # Dilation method:
            #          bin_ind_di = binary_dilation(bin_ind, iterations=2)
            #          mask_subset[bin_ind_di] = ind
            #          mask_subset[bin_ind] = 0
            interested_voxels[mask_subset != 0] = True
            if fullaparc:
                FS_LUT = pd.read_csv('/home/jagust/petcore/qc/scripts/required/FreeSurferColorLUT_brain.txt')#.set_index('Index')
                FS_LUT[['R','G','B','A']] = FS_LUT[['R','G','B','A']]#/256
                FS_LUT['A'] = [255,]*len(FS_LUT['A'])
                FS_LUT.loc[0,:]= [0,'Unknown',0,0,0,0]
                FS_LUT.loc[len(FS_LUT)] = [999999,'Error',0,0,0,0]
                FS_dict = FS_LUT[['Index','R','G','B','A']].set_index('Index').astype(int).T.to_dict('list')
                u,inv = np.unique(mask_subset,return_inverse = True)
                mask_subset = np.array([FS_dict[x] for x in u])[inv].reshape(mask_subset.shape+(4,))
            else:    
                # Cut down the LUT to just the ROI we want to use
                LUT_subset = Color_LUT.loc[np.isin(Color_LUT['Name'] , [roi_name,'Unknown'])]
                FS_dict = LUT_subset[['Index','R','G','B','A']].set_index('Index').astype(int).T.to_dict('list')
                u,inv = np.unique(mask_subset.astype(int),return_inverse = True)
                mask_subset = np.array([FS_dict[x] for x in u])[inv].reshape(mask_subset.shape+(4,))
                #mask_subset[:,:,:,3] = mask_subset[:,:,:,3] * .5 #manual alpha

            # These images are 4d and ready to be converted to PIL
            if show_mri:
                display_numpy_images.append(mask_subset)
                erode_values.append(row['erode'])
            if show_pet:
                display_numpy_images_pet.append(mask_subset)
                erode_values_pet.append(row['erode'])


    # Create reference region
    REF_MASK = np.zeros(NU.shape, dtype=bool)
    prev_mask = np.nan
    if inorm:
        for mask in reference['fullmask'].unique():
            rois = list(reference.loc[reference['fullmask'] == mask]['value'])
            if prev_mask != mask:
                print('Loading (reference):',mask)
                mask_data = np.round(np.asarray(nib.load(mask).get_fdata()),1)
            REF_MASK[np.isin(mask_data,rois)] = True
            # Don't reload the mask if we already have it loaded.
            prev_mask = mask

        if (REF_MASK==False).all():
            print('Warning: REF_MASK is empty. ')

    # Load and normalize the PET to 0-255
    PET = np.asarray(nib.load(pet_path).get_fdata(),dtype=np.float32)
    print(PET.shape)
    print(REF_MASK.shape)
    if inorm:
        ref_mean = np.nanmean(PET[REF_MASK])
        PET = PET / ref_mean

    return (PET,NU,interested_voxels,display_numpy_images,display_numpy_images_pet,erode_values,erode_values_pet)


def create_metadata_dict(nu_path,pet_path,display_df,reference,inorm, orig=False):
    display_df['fullmask'] = (display_df['folder'] +'/'+ display_df['mask'])
    # Gets the intensity normalized PET (if inorm == True)

    if orig:
        nu_path,pet_path,display_df,reference = orig

    PET,NU,interested_voxels,display_numpy_images,display_numpy_images_pet,eR,eRpet = loadReqData(nu_path, pet_path, display_df,reference,inorm)

    meta_dict = {}

    meta_dict['NU_img']=os.path.realpath(nu_path)
    meta_dict['PET_img']=os.path.realpath(pet_path)

    pet_mod_time = time.ctime(os.path.getmtime(pet_path))
    mri_mod_time = time.ctime(os.path.getmtime(nu_path))

    meta_dict['PET_modify_date'] = pet_mod_time
    meta_dict['MRI_modify_date'] = mri_mod_time
    for mask_path in np.unique(list(display_df['fullmask'].values)):
        mask_dat = np.nan_to_num(nib.load(mask_path).get_fdata())
        if 'aparc+aseg' in os.path.basename(mask_path):mask_indencies=QC_cortex_codes
        else:mask_indencies = np.unique(mask_dat)
        mask_indencies = np.asarray(mask_indencies).astype(int)
        if len(mask_indencies) > 2000:
            raise ValueError('ERROR Error error - There is a mask with more than 1000 unique values!!! This is not likely a mask...')
        for ivalue in mask_indencies:
            if (ivalue == 0):
                continue
            mask_pathb = os.path.basename(mask_path)
            if mask_pathb[:2] == 'a_':
                mask_pathb = mask_pathb[2:]
            meta_suvr_key = f'{os.path.basename(mask_pathb)} {ivalue:.0f} SUVR'
            meta_size_key = f'{os.path.basename(mask_pathb)} {ivalue:.0f} SIZE'
            meta_suvr = np.nanmean(PET[mask_dat == ivalue])
            meta_size = np.nansum(mask_dat == ivalue)
            meta_dict[meta_suvr_key] = f'{meta_suvr:.3f}'
            meta_dict[meta_size_key] = f'{meta_size:.0f}'
    return meta_dict

def update_metadata_dict(pngimage, nu_path, pet_path, display_df, reference, inorm):
    metadata = PngInfo()
    image_meta = create_metadata_dict(nu_path,pet_path,display_df,reference,inorm)
    for dkey in list(image_meta.keys()):
        metadata.add_text(dkey, image_meta[dkey])

    print('Saving QC image:',pngimage)
    img = Image.open(pngimage)
    img.save(pngimage,pnginfo=metadata,optimize=True)


def check_img(nu_path, pet_path, img_output,display_df,reference, alt_path_names,timepoint, VMIN,VMAX,inorm):
    '''
     Concept: All the input to the draw func should be added to the QC image meta and check for consistency when creating new QC image.
     Add SUVR for all FS regions in aparc+aseg. Rerun if SUVR varies too much between SUVR or VOLUME. R2 < 0.99
     Do not use recon-date as we want subsequent processing to not trigger new QC image if the image does not change.

     Return True if image is good. Return False if image must be re-created.
    '''
    print('---------- Check Data -------------')
    print('\n',nu_path,'\n',img_output,'\n',timepoint,'\n','\n',pet_path,'\n',f'MIN/MAX: {VMIN}-{VMAX}')
    MRI_date, PET_date = timepoint
    if not os.path.exists(img_output):
        print('image_output does not exist:',img_output)
        print('check_img returning False')
        return False
    try:
        existing_image_meta = Image.open(img_output).text
        try:meta_MRI_date = re.search(r'MRI_(\d{4}-\d{2}-\d{2})', existing_image_meta['NU_img']).group(1)
        except:meta_MRI_date = 'MRI-DATE-FAILED'
        
        if str(meta_MRI_date) != str(MRI_date):
            # The MRI tp has changed. Always reprocess so that MRI date in QC image is correct. 
            print('MRI date has changed in this analysis. Rerunning QC image.')
            return False

        expecting_image_meta = create_metadata_dict(nu_path,pet_path,display_df,reference,inorm)

        x_size=[]
        y_size=[]
        y_suvr=[]
        x_suvr=[]

        for dkey in list(expecting_image_meta.keys()):
            A=expecting_image_meta[dkey]

            # This is some legacy code. When I first ran the, the metadata ROI indices were 
            #   sometimes ints, sometimes floats. Sometimes aligned 'a_', sometimes not. 
            #  I fixed the code to only use int but a_ may still happen.
            dkey2 = 'a_'+dkey.replace(' S','.0 S')
            dkey3 = 'a_'+dkey
            dkey4 = dkey.replace(' S','.0 S')
            if dkey in list(existing_image_meta.keys()): 
                B=existing_image_meta[dkey]
            elif dkey2 in list(existing_image_meta.keys()): 
                B=existing_image_meta[dkey2]
            elif dkey3 in list(existing_image_meta.keys()): 
                B=existing_image_meta[dkey3]
            elif dkey4 in list(existing_image_meta.keys()): 
                B=existing_image_meta[dkey4]

            else:
                print('********** BAD dkey *************')
                print('Existing:')
                print(list(existing_image_meta.keys())[:10])
                print('Expecting:')
                print(list(expecting_image_meta.keys())[:10])
                print('')
                print([dkey,dkey2,dkey3])
                
                time.sleep(5)
            
            if A.replace('.','',1).isdigit() and B.replace('.','',1).isdigit():
                A,B=float(A),float(B)
                if dkey[-4:] == 'SIZE':
                    x_size.append(A)
                    y_size.append(B)
                if dkey[-4:] == 'SUVR':
                    x_suvr.append(A)
                    y_suvr.append(B)

        #print(x_size,y_size,x_suvr,y_suvr)
        rsquare_size=r2_score(np.nan_to_num(x_size),np.nan_to_num(y_size))
        rsquare_suvr=r2_score(np.nan_to_num(x_suvr),np.nan_to_num(y_suvr))

        suvr_diff = np.nan_to_num(x_suvr) - np.nan_to_num(y_suvr)
        print(suvr_diff)
        if np.nanmax(suvr_diff) > 0.1:
            print('SUVR difference =',np.nanmax(suvr_diff),'\nRerunning QC image. Archiving data.')
            iji = list(suvr_diff).index(np.nanmax(suvr_diff))
            print(list(expecting_image_meta.keys())[iji])
            print(np.nan_to_num(x_suvr)[iji],np.nan_to_num(y_suvr)[iji])
            ''' If there is a greater SUVR difference than 1/10th SUVR, rerun QC image.'''
            return False

        x_suvr = np.asarray(x_suvr).reshape((-1,1))
        y_suvr = np.asarray(y_suvr).reshape((-1,1))

        lm=LinearRegression()
        lm.fit(x_suvr,y_suvr)
        Yhat = lm.predict(x_suvr)


        print('Number of values used in correlation:',len(x_size),len(y_suvr))
        print('SUVR correlation:',rsquare_suvr)
        print('VOLUME correlation:',rsquare_size)
        print('SUVR slope:',lm.coef_)
        print('SUVR y-intercept:',lm.intercept_)

        if (rsquare_size >= 0.99) and (rsquare_suvr >= 0.99) and (np.round(lm.intercept_[0],2) == 0) and (np.round(lm.coef_[0][0],2) == 1):
            print('All looks good... Not rerunning.')
            return True
        else:
            print('Correlation is not good enough. Rerunning.')
            return False

    except Exception as e:
        print("************* EXCEPTION ***************")
        print(e)
        print("***************************************")
        return False
    return False


def draw_image(nu_path, pet_path, output,display_df:pd.DataFrame,reference:pd.DataFrame,alt_path_names=False,timepoint=False, VMIN: float = 0.5,VMAX: float = 2.5,inorm=False,cols=12):
    '''
     Primary function for QC image creation

     INPUT:
     nu_path: path to nu.nii (3d)
     display_df: pandas dataframe specifying ROIs to display
         Uses LUT to determine color. White if the name DNE.
     reference: pandas dataframe specifying reference region stuff
     output: output filepath ending in png or jpg
     pet_path: path of pet image
     timepoint: False or string denoting the Timepoint of the image BL, Scan2, etc.
     VMIN,VMAX: Visual thresholds to use for PET colormap.
     inorm: Do you want to apply i-normalization to PET image?

     OUTPUT:
     Returns True if success
     Saves image to output path (png or jpg)
    '''
    print('Running draw_image_PIL:',nu_path, pet_path)
    #from scipy.ndimage import binary_erosion
    print('\n',nu_path,'\n',output,'\n',timepoint,'\n','\n',pet_path,'\n',f'MIN/MAX: {VMIN}-{VMAX}')


    PET,NU,interested_voxels,display_numpy_images,display_numpy_images_pet,erode_values,erode_values_pet = loadReqData(nu_path, pet_path, display_df,reference,inorm)

    ####
    # Clipping. At no point past this comment should we consider the data for quantification such as getting cortical summary SUVRs.
    # BELOW PET DATA DO NOT TRUST IT.
    ####
    PET[PET < VMIN]=np.nan
    PET[PET > VMAX]=np.nan
    PET = (PET - VMIN)/(VMAX - VMIN)
    PET[PET<=0.0001] = np.nan
    PET_colored = NIH(PET, bytes=True)
    PET_colored[:,:,:,3] = PET_colored[:,:,:,3]*0.6

    NU_colored = cm.gray(NU, bytes=True)

    # How many columns of images to use:
    Col_Num = cols

    # Axial
    L_vols = np.sum(interested_voxels,axis=(0,1))
    mean_vol = np.std(L_vols[~np.equal(L_vols,0)])
    applicable_indexes = np.asarray([i for i in range(interested_voxels.shape[2]) if np.sum(interested_voxels[:,:,i]) > mean_vol])
    Axial_Slices = np.sort(applicable_indexes[np.linspace(0,len(applicable_indexes)-1, Col_Num+5,dtype=int)])[2:-3]

    # Sagittal
    L_vols = np.sum(interested_voxels,axis=(1,2))
    mean_vol = np.std(L_vols[~np.equal(L_vols,0)])
    applicable_indexes = np.asarray([i for i in range(interested_voxels.shape[0]) if np.sum(interested_voxels[i,:,:]) > mean_vol])
    Sagittal_Slices = np.sort(applicable_indexes[np.linspace(0,len(applicable_indexes)-1, Col_Num+4,dtype=int)])[2:-2]

    # Coronal
    L_vols = np.sum(interested_voxels,axis=(0,2))
    mean_vol = np.std(L_vols[~np.equal(L_vols,0)])
    applicable_indexes = np.asarray([i for i in range(interested_voxels.shape[1]) if np.sum(interested_voxels[:,i,:]) > mean_vol])
    Coronal_Slices = np.sort(applicable_indexes[np.linspace(0,len(applicable_indexes)-1, Col_Num+4,dtype=int)])[2:-2]


    #bordertop = cm.gray(np.repeat(np.zeros(1500).reshape(1,-1),40,0),bytes=True)
    #sideborder = cm.gray(np.repeat(np.zeros(150).reshape(-1,1),40,1),bytes=True)
    stst_int = [i for i in range(interested_voxels.shape[0]) if (False in (interested_voxels[i,:,:] == 0))]
    start_0 = np.min(stst_int)
    stop_0 = np.max(stst_int)
    stst_int = [i for i in range(interested_voxels.shape[1]) if (False in (interested_voxels[:,i,:] == 0))]
    start_1 = np.min(stst_int)
    stop_1 = np.max(stst_int)
    stst_int = [i for i in range(interested_voxels.shape[2]) if (False in (interested_voxels[:,:,i] == 0))]
    start_2 = np.min(stst_int)
    stop_2 = np.max(stst_int)


    start_x = np.min([start_0,start_1,start_2])
    stop_x = np.max([stop_0,stop_1,stop_2])

    print('MR images')
    # Below can be used for S or C slices

    ax_imgs_raw = []
    for S in Axial_Slices:
        ax_imgs_raw.append(prep_img([start_0, stop_0, start_x, stop_x],S,np.copy(NU_colored)[:,:,S,:],False,False,erode_values))
    ax_imgs_raw = np.concatenate(ax_imgs_raw,axis=1)

    #scan_summary = np.concatenate([ax_imgs_raw,ax_imgs,cor_imgs,sag_imgs],axis=0)
    scan_summary = np.concatenate([ax_imgs_raw,],axis=0)


    if len(display_numpy_images) > 0:
        ax_imgs = []
        for S in Axial_Slices:
            #display_numpy_images_sliced = [img[:,:,S,:] for img in display_numpy_images]
            ax_imgs.append(prep_img([start_0, stop_0, start_1, stop_1],S,np.copy(NU_colored)[:,:,S,:],False, [np.copy(i)[:,:,S,:] for i in display_numpy_images],erode_values))
        ax_imgs = np.concatenate(ax_imgs,axis=1)

        cor_imgs = []
        for S in Coronal_Slices:
            #display_numpy_images_sliced = [img[:,S,:,:] for img in display_numpy_images]
            cor_imgs.append(prep_img([start_0, stop_0, start_2, stop_2],S,np.copy(NU_colored)[:,S,:,:],False, [np.copy(i)[:,S,:,:] for i in display_numpy_images],erode_values ))
        cor_imgs = np.concatenate(cor_imgs,axis=1)

        sag_imgs = []
        for S in Sagittal_Slices:
            #display_numpy_images_sliced = [img[S,:,:,:] for img in display_numpy_images]
            sag_imgs.append(prep_img([start_1, stop_1, start_2, stop_2],S,np.copy(NU_colored)[S,:,:,:],False,[np.copy(i)[S,:,:,:] for i in display_numpy_images],erode_values))
        sag_imgs = np.concatenate(sag_imgs,axis=1)

        scan_summary = np.concatenate([scan_summary,ax_imgs,cor_imgs,sag_imgs],axis=0)
        #scan_summary = np.concatenate([scan_summary,ax_imgs,cor_imgs,sag_imgs],axis=0)

    # PET Images
    if pet_path:
        ax_imgs = []
        for S in Axial_Slices:
            ax_imgs.append(prep_img([start_0, stop_0, start_1, stop_1],S,np.copy(NU_colored)[:,:,S,:],np.copy(PET_colored)[:,:,S,:], [np.copy(i)[:,:,S,:] for i in display_numpy_images_pet],erode_values_pet))
        ax_imgs = np.concatenate(ax_imgs,axis=1)

        cor_imgs = []
        for S in Coronal_Slices:
            cor_imgs.append(prep_img([start_0, stop_0, start_2, stop_2],S,np.copy(NU_colored)[:,S,:,:],np.copy(PET_colored)[:,S,:,:], [np.copy(i)[:,S,:,:] for i in display_numpy_images_pet],erode_values_pet))
        cor_imgs = np.concatenate(cor_imgs,axis=1)

        sag_imgs = []
        for S in Sagittal_Slices:
            sag_imgs.append(prep_img([start_1, stop_1, start_2, stop_2],S,np.copy(NU_colored)[S,:,:,:],np.copy(PET_colored)[S,:,:,:],[np.copy(i)[S,:,:,:] for i in display_numpy_images_pet],erode_values_pet))
        sag_imgs = np.concatenate(sag_imgs,axis=1)

        scan_summary = np.concatenate([scan_summary,ax_imgs,cor_imgs,sag_imgs],axis=0)

    # Add colorbar to bottom
    cbar_size = scan_summary.shape[1] - 400

    cbar_height = 30
    cbar = NIH(np.repeat(np.linspace(0,1,cbar_size).reshape(1,-1),cbar_height,0), bytes=True)
    cbar[0:1,0:,:] = [255,255,255,255]
    cbar[-1:,0:,:] = [255,255,255,255]
    cbar[:,0,:] = [255,255,255,255]
    cbar[:,-1,:] = [255,255,255,255]
    border = cm.gray(np.repeat(np.zeros(scan_summary.shape[1]).reshape(1,-1),30,0),bytes=True)
    bordertop = cm.gray(np.repeat(np.zeros(scan_summary.shape[1]).reshape(1,-1),15,0),bytes=True)
    sideborder = cm.gray(np.repeat(np.zeros(cbar_height).reshape(-1,1),200,1),bytes=True)
    #whitesideborder = cm.gray(np.repeat(np.ones(70).reshape(-1,1),20,1),bytes=True)


    fnt = ImageFont.truetype(font_path, 30)
    cbar = np.hstack((cbar,sideborder))
    cbar = np.hstack((sideborder,cbar))
    cbar = np.vstack((bordertop,cbar))
    cbar = np.vstack((cbar,border))
    #cbar = np.hstack((cbar,sideborder))
    #cbar = np.hstack((sideborder,cbar))
    cbar = Image.fromarray(cbar)
    d = ImageDraw.Draw(cbar)
    d.text((170,15), str(VMIN), font=fnt, fill=(255,255,255,255))
    d.text((cbar.size[0]-180,15), '≥'+str(VMAX), font=fnt, fill=(255,255,255,255))
    cbar = np.asarray(cbar)

    scan_summary = np.vstack((scan_summary,cbar))

    # Add on summary info and sidebars
    border = cm.gray(np.repeat(np.zeros(20).reshape(1,-1),scan_summary.shape[0],0),bytes=True)
    scan_summary = np.hstack((border,scan_summary,border))
    border = cm.gray(np.repeat(np.zeros(scan_summary.shape[1]).reshape(1,-1),100,0),bytes=True)

    border = Image.fromarray(border)
    d = ImageDraw.Draw(border)
    fnt = ImageFont.truetype(font_path, 30)
    d.text((20,15), str(alt_path_names[0]), font=fnt, fill=(255,255,255,255))
    d.text((20,55), str(alt_path_names[1]), font=fnt, fill=(255,255,255,255))
    d.text((border.size[0]-340,15), f'MRI date: {timepoint[0]}', font=fnt, fill=(255,255,255,255))
    d.text((border.size[0]-340,55), f'PET date: {timepoint[1]}', font=fnt, fill=(255,255,255,255))
    #fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 30)
    #d.text((20,65), str(scan_string), font=fnt, fill=(255,255,255,255))
    border = np.asarray(border)


    scan_summary = np.vstack((border,scan_summary))

    try:
        scan_summary=Image.fromarray(scan_summary)
        scan_summary = scan_summary.convert('RGB')
        metadata = PngInfo()
        image_meta = create_metadata_dict(nu_path,pet_path,display_df,reference,inorm,orig=alt_path_names)
        for dkey in list(image_meta.keys()):
            metadata.add_text(dkey, image_meta[dkey])

        print('Saving QC image:',output)
        scan_summary.save(output,pnginfo=metadata,optimize=True)

    except:
        print('** ERROR: Could not save image',output)

    return True

