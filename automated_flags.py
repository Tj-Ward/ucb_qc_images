#!/usr/bin/env python
#import subprocess
#import nibabel as nib
#import os
#import re
#import sys
#from glob import glob
#from datetime import date as Date
#from datetime import datetime
#import pandas as pd
#import numpy as np

ALL_ROIS = ['BRAIN_STEM',
        'BRAIN_STEM_SIZE',
        'CTX_LH_CAUDALANTERIORCINGULATE',
        'CTX_LH_CAUDALANTERIORCINGULATE_SIZE',
        'CTX_LH_CAUDALMIDDLEFRONTAL',
        'CTX_LH_CAUDALMIDDLEFRONTAL_SIZE',
        'CTX_LH_CUNEUS',
        'CTX_LH_CUNEUS_SIZE',
        'CTX_LH_ENTORHINAL',
        'CTX_LH_ENTORHINAL_SIZE',
        'CTX_LH_FRONTALPOLE',
        'CTX_LH_FRONTALPOLE_SIZE',
        'CTX_LH_FUSIFORM',
        'CTX_LH_FUSIFORM_SIZE',
        'CTX_LH_INFERIORPARIETAL',
        'CTX_LH_INFERIORPARIETAL_SIZE',
        'CTX_LH_INFERIORTEMPORAL',
        'CTX_LH_INFERIORTEMPORAL_SIZE',
        'CTX_LH_INSULA',
        'CTX_LH_INSULA_SIZE',
        'CTX_LH_ISTHMUSCINGULATE',
        'CTX_LH_ISTHMUSCINGULATE_SIZE',
        'CTX_LH_LATERALOCCIPITAL',
        'CTX_LH_LATERALOCCIPITAL_SIZE',
        'CTX_LH_LATERALORBITOFRONTAL',
        'CTX_LH_LATERALORBITOFRONTAL_SIZE',
        'CTX_LH_LINGUAL',
        'CTX_LH_LINGUAL_SIZE',
        'CTX_LH_MEDIALORBITOFRONTAL',
        'CTX_LH_MEDIALORBITOFRONTAL_SIZE',
        'CTX_LH_MIDDLETEMPORAL',
        'CTX_LH_MIDDLETEMPORAL_SIZE',
        'CTX_LH_PARACENTRAL',
        'CTX_LH_PARACENTRAL_SIZE',
        'CTX_LH_PARAHIPPOCAMPAL',
        'CTX_LH_PARAHIPPOCAMPAL_SIZE',
        'CTX_LH_PARSOPERCULARIS',
        'CTX_LH_PARSOPERCULARIS_SIZE',
        'CTX_LH_PARSORBITALIS',
        'CTX_LH_PARSORBITALIS_SIZE',
        'CTX_LH_PARSTRIANGULARIS',
        'CTX_LH_PARSTRIANGULARIS_SIZE',
        'CTX_LH_PERICALCARINE',
        'CTX_LH_PERICALCARINE_SIZE',
        'CTX_LH_POSTCENTRAL',
        'CTX_LH_POSTCENTRAL_SIZE',
        'CTX_LH_POSTERIORCINGULATE',
        'CTX_LH_POSTERIORCINGULATE_SIZE',
        'CTX_LH_PRECENTRAL',
        'CTX_LH_PRECENTRAL_SIZE',
        'CTX_LH_PRECUNEUS',
        'CTX_LH_PRECUNEUS_SIZE',
        'CTX_LH_ROSTRALANTERIORCINGULATE',
        'CTX_LH_ROSTRALANTERIORCINGULATE_SIZE',
        'CTX_LH_ROSTRALMIDDLEFRONTAL',
        'CTX_LH_ROSTRALMIDDLEFRONTAL_SIZE',
        'CTX_LH_SUPERIORFRONTAL',
        'CTX_LH_SUPERIORFRONTAL_SIZE',
        'CTX_LH_SUPERIORPARIETAL',
        'CTX_LH_SUPERIORPARIETAL_SIZE',
        'CTX_LH_SUPERIORTEMPORAL',
        'CTX_LH_SUPERIORTEMPORAL_SIZE',
        'CTX_LH_SUPRAMARGINAL',
        'CTX_LH_SUPRAMARGINAL_SIZE',
        'CTX_LH_TEMPORALPOLE',
        'CTX_LH_TEMPORALPOLE_SIZE',
        'CTX_LH_TRANSVERSETEMPORAL',
        'CTX_LH_TRANSVERSETEMPORAL_SIZE',
        'CTX_RH_CAUDALANTERIORCINGULATE',
        'CTX_RH_CAUDALANTERIORCINGULATE_SIZE',
        'CTX_RH_CAUDALMIDDLEFRONTAL',
        'CTX_RH_CAUDALMIDDLEFRONTAL_SIZE',
        'CTX_RH_CUNEUS',
        'CTX_RH_CUNEUS_SIZE',
        'CTX_RH_ENTORHINAL',
        'CTX_RH_ENTORHINAL_SIZE',
        'CTX_RH_FRONTALPOLE',
        'CTX_RH_FRONTALPOLE_SIZE',
        'CTX_RH_FUSIFORM',
        'CTX_RH_FUSIFORM_SIZE',
        'CTX_RH_INFERIORPARIETAL',
        'CTX_RH_INFERIORPARIETAL_SIZE',
        'CTX_RH_INFERIORTEMPORAL',
        'CTX_RH_INFERIORTEMPORAL_SIZE',
        'CTX_RH_INSULA',
        'CTX_RH_INSULA_SIZE',
        'CTX_RH_ISTHMUSCINGULATE',
        'CTX_RH_ISTHMUSCINGULATE_SIZE',
        'CTX_RH_LATERALOCCIPITAL',
        'CTX_RH_LATERALOCCIPITAL_SIZE',
        'CTX_RH_LATERALORBITOFRONTAL',
        'CTX_RH_LATERALORBITOFRONTAL_SIZE',
        'CTX_RH_LINGUAL',
        'CTX_RH_LINGUAL_SIZE',
        'CTX_RH_MEDIALORBITOFRONTAL',
        'CTX_RH_MEDIALORBITOFRONTAL_SIZE',
        'CTX_RH_MIDDLETEMPORAL',
        'CTX_RH_MIDDLETEMPORAL_SIZE',
        'CTX_RH_PARACENTRAL',
        'CTX_RH_PARACENTRAL_SIZE',
        'CTX_RH_PARAHIPPOCAMPAL',
        'CTX_RH_PARAHIPPOCAMPAL_SIZE',
        'CTX_RH_PARSOPERCULARIS',
        'CTX_RH_PARSOPERCULARIS_SIZE',
        'CTX_RH_PARSORBITALIS',
        'CTX_RH_PARSORBITALIS_SIZE',
        'CTX_RH_PARSTRIANGULARIS',
        'CTX_RH_PARSTRIANGULARIS_SIZE',
        'CTX_RH_PERICALCARINE',
        'CTX_RH_PERICALCARINE_SIZE',
        'CTX_RH_POSTCENTRAL',
        'CTX_RH_POSTCENTRAL_SIZE',
        'CTX_RH_POSTERIORCINGULATE',
        'CTX_RH_POSTERIORCINGULATE_SIZE',
        'CTX_RH_PRECENTRAL',
        'CTX_RH_PRECENTRAL_SIZE',
        'CTX_RH_PRECUNEUS',
        'CTX_RH_PRECUNEUS_SIZE',
        'CTX_RH_ROSTRALANTERIORCINGULATE',
        'CTX_RH_ROSTRALANTERIORCINGULATE_SIZE',
        'CTX_RH_ROSTRALMIDDLEFRONTAL',
        'CTX_RH_ROSTRALMIDDLEFRONTAL_SIZE',
        'CTX_RH_SUPERIORFRONTAL',
        'CTX_RH_SUPERIORFRONTAL_SIZE',
        'CTX_RH_SUPERIORPARIETAL',
        'CTX_RH_SUPERIORPARIETAL_SIZE',
        'CTX_RH_SUPERIORTEMPORAL',
        'CTX_RH_SUPERIORTEMPORAL_SIZE',
        'CTX_RH_SUPRAMARGINAL',
        'CTX_RH_SUPRAMARGINAL_SIZE',
        'CTX_RH_TEMPORALPOLE',
        'CTX_RH_TEMPORALPOLE_SIZE',
        'CTX_RH_TRANSVERSETEMPORAL',
        'CTX_RH_TRANSVERSETEMPORAL_SIZE',
        'WM_HYPOINTENSITIES',
        'WM_HYPOINTENSITIES_SIZE']

def MovingMeanSD(loni_path,qc_df):
    import importlib.util

    #loni_path = '/home/jagust/adni/pipeline_scripts/csv-maker/output/10-20-2020/FS7_UCBERKELEYAV1451_10-20-2020_regular_tp.csv'
   
    loni = pd.read_csv(loni_path)
    loni['RID'] = loni['RID'].astype(int).astype(str)
    qc_df=qc_df.reset_index(drop=False)

    loni['PET_Date']=np.nan
    # print('loni pet date',loni['d_visit'])
    # loni['PET_DateStr']=list([datetime.strptime(_,'%m/%d/%y') for _ in loni['d_visit'].astype(str)])
    # print('qc PET DATE',qc_df['PET_DateStr'])
    # loni['PET_Date']=list([datetime.strftime(_,'%Y-%m-%d') for _ in loni['PET_DateStr']])
    # print('qc PET DATE',qc_df['PET_Date'])
    loni['MRI_Date']=np.nan
    
    for _ in loni.index:
        rid=loni.loc[_,'RID']
        try:
            tp=loni.loc[_,'TP']
        except:
            tp='BL'
        mdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'MRI_Date']
        pdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'PET_Date']
        if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))!=1:
            if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))==0:
                mdate=''
                pdate=''
            else:  
                mdate=sorted(list(mdate),reverse=True)[0]
                pdate=sorted(list(pdate),reverse=True)[0]
        else:
            mdate=list(mdate)[0]
            pdate=list(pdate)[0]
        loni.loc[_,'PET_Date']=pdate
        loni.loc[_,'MRI_Date']=mdate

    loni = loni.set_index(['RID','TP','MRI_Date','PET_Date'])
    idx_to_drop=qc_df.loc[qc_df.Usability==0,['RID','TP','MRI_Date','PET_Date']]
    idx_to_drop=np.unique(idx_to_drop)
    # print('drop',idx_to_drop)
    loni_train=loni.drop(idx_to_drop)
    
    spec = importlib.util.spec_from_file_location("searchList", '/home/jagust/tjward/Notebooks/Tyler-Github/Data_Finder/tjw_search_tools.py')
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    
    # COLS_OF_INTEREST = (loni.drop(['VISCODE','VISCODE2','EXAMDATE','INJTIME','SCANTIME'],axis=1).dropna(axis=1).columns)
    print('..')
    VOL_COLS = st.searchList(ALL_ROIS,Include=['SIZE'],Exclude=['SUVR','AUTO_FLAG'],return_list=1)[0]
    VOL_RANGE =[]
    SUVR_COLS = st.searchList(ALL_ROIS,Include=[],Exclude=['SIZE','AUTO_FLAG'],return_list=1)[0]
    SUVR_RANGE =[]
    print('...')
    for VCOL in VOL_COLS:
        # SUVR Flags
        vols=loni_train[VCOL]
        meanSDlo = np.nanmean(vols) - np.nanstd(vols) - np.nanstd(vols)
        meanSDhi = np.nanmean(vols) + np.nanstd(vols) + np.nanstd(vols)
        VOL_RANGE.append([meanSDlo,meanSDhi])
    for _ in loni.index:
        subj_bool=[]
        for x in range(len(VOL_COLS)):
            # SUVR Flags
            VCOL=VOL_COLS[x]
            lo=VOL_RANGE[x][0]
            hi=VOL_RANGE[x][1]
            vol=loni.loc[_,VCOL]
            outlier=int(((vol<lo)|(vol>hi)))
            subj_bool.append(outlier)
        flag=int(sum(subj_bool)>0)
        loni.loc[_,'AUTO_FLAG_VOL'] = flag
    print('....')
    for SCOL in SUVR_COLS:
        # VOL QCs
        suvrs=loni_train[SCOL]
        meanSDlo = np.nanmean(suvrs) - np.nanstd(suvrs) - np.nanstd(suvrs)
        meanSDhi = np.nanmean(suvrs) + np.nanstd(suvrs) + np.nanstd(suvrs)
        SUVR_RANGE.append([meanSDlo,meanSDhi])
    for _ in loni.index:
        subj_bool=[]
        for x in range(len(VOL_COLS)):
            # SUVR Flags
            SCOL=SUVR_COLS[x]
            lo=SUVR_RANGE[x][0]
            hi=SUVR_RANGE[x][1]
            vol=loni.loc[_,SCOL]
            outlier=int(((vol<lo)|(vol>hi)))
            subj_bool.append(outlier)
        flag=int(sum(subj_bool)>0)
        loni.loc[_,'AUTO_FLAG_SUVR'] = flag

    idx = loni.index
    loni.index = loni.index.set_levels([idx.levels[0].astype(int), idx.levels[1].astype(str), idx.levels[2].astype(str), idx.levels[3].astype(str)])
    return (loni['AUTO_FLAG_SUVR'], loni['AUTO_FLAG_VOL'])


def LR_symmetry(aparc_path):
    import nibabel as nib
    import numpy as np
    import os

    if not os.path.exists(aparc_path):
        return np.nan

    aparc_dat = nib.load(aparc_path).get_fdata()
    LH = np.logical_and(aparc_dat >= 1000, aparc_dat < 2000)
    RH = np.logical_and(aparc_dat >= 2000, aparc_dat < 3000)
    BOTH = np.logical_or(LH,RH)

    Val = (abs(np.sum(LH) - np.sum(RH)) / np.sum(BOTH))
    return f'{Val:.4f}'

def cereb_GM_overlap(mri_path):
    import nibabel as nib
    import numpy as np
    import scipy.ndimage as ndimage

    aparc_path = os.path.join(mri_path,'raparc+aseg.nii')
    spm_gm_path = os.path.join(mri_path,'c1rnu.nii')
    spm_wm_path = os.path.join(mri_path,'c2rnu.nii')

    check_paths_list = [aparc_path,spm_gm_path,spm_wm_path]
    for p in check_paths_list:
        if not os.path.exists(p):
            return np.nan

    aparc_img = nib.load(aparc_path)
    spm_gm_img = nib.load(spm_gm_path)
    spm_wm_im = nib.load(spm_wm_path)

    aparc_data = np.round(aparc_img.get_fdata(),0)
    spm_gm_mask = spm_gm_img.get_fdata()
    spm_wm_mask = spm_gm_img.get_fdata()

    spm_brain_mask = spm_gm_mask + spm_wm_mask >= 0.1
    TIV = np.nansum(spm_brain_mask)

    fwhm=10
    fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
    sigma = fwhm / (fwhm_over_sigma_ratio * 1)
    
    FS_WC = np.isin(aparc_data,[7,8,46,47]).astype(float)
    FS_WC_smoo = ndimage.gaussian_filter(FS_WC, sigma=sigma, mode='constant', cval=0)
    FS_brainmask = (aparc_data != 0).astype(float)
    FS_brainmask_smoo = ndimage.gaussian_filter(FS_brainmask, sigma=sigma, mode='constant', cval=0)
    
    spm_brain_mask[FS_brainmask_smoo >= 0.1] = 0
    spm_brain_mask[FS_WC_smoo >= 0.1] = 0

    missing_cereb = (spm_brain_mask / TIV)*100
    return missing_cereb




def pet_bounding_box(aparc_path, pet_path):
    import nibabel as nib
    import numpy as np
    import os
    '''
    Returns percentage of cerebrum and cerebellum overlap with PET bounding box as a tuple,

    ( % cerebrum , % cerebellum)

    '''
    if os.path.exists(pet_path):
        PET_DAT = nib.load(pet_path).get_fdata()
        PET_DAT[PET_DAT <= 0.05] = np.nan
    else:
        print('***ERROR***')
        print('PET path invalid:',pet_path)
        print('Skipping...')
        return (np.nan,np.nan)

    if os.path.exists(aparc_path):
        ASEG_DAT = nib.load(aparc_path).get_fdata()
    else:
        print('***ERROR***')
        print('APARC path invalid:',aparc_path)
        print('Skipping...')
        return (np.nan,np.nan)

    # Brain ROI is anything not 0, exclude Brainstem (16), and cerebellum (8,7,46,47)
    index_arr_cerebrum =  np.logical_not(np.isin(ASEG_DAT, [0,16,8,7,46,47]))
    index_arr_cerebellum = np.isin(ASEG_DAT, [8,7,46,47])

    # Determine if PET data is NaN at in brain
    sum_cerebrum_nam = np.sum(np.isnan(PET_DAT[index_arr_cerebrum]))
    sum_cerebellum_nan = np.sum(np.isnan(PET_DAT[index_arr_cerebellum]))

    # Size of rois
    size_cerebrum = np.sum(index_arr_cerebrum)
    size_cerebellum = np.sum(index_arr_cerebellum)

    percent_cerebrum_nan = 1-np.round(sum_cerebrum_nam / size_cerebrum,5)
    percent_cerebrum_nan = f'{100*percent_cerebrum_nan:.1f}%'
    percent_cerebellum_nan = 1-np.round(sum_cerebellum_nan / size_cerebellum,5)
    percent_cerebellum_nan = f'{100*percent_cerebellum_nan:.1f}%'
    # print(percent_cerebrum_nan, percent_cerebellum_nan)
    return (percent_cerebrum_nan, percent_cerebellum_nan)


def image_correlation(src,ref):
    '''
    This must return something. Whitespace will be replaced with nan.
    Nan values in QC CSV will be read as not having been ran, rerunning autoQC every week.
    Currently returns 1 if BL scan. Could also return a string like BL or "."
    '''
    from scipy.stats import pearsonr
    import numpy as np
    import nibabel as nib
    if src == ref:
        return 1
    A = np.nan_to_num(nib.load(src).get_fdata())
    B = np.nan_to_num(nib.load(ref).get_fdata())
    A=A.reshape(-1)
    B=B.reshape(-1)
    try:
        r,p = pearsonr(A,B)
        score=r
        score = f'{score:.4f}'
    except Exception as e:
        print(e)
        print('Exception in image correlation')
        score = np.nan
    return (score)

#def OneClassSVM(loni_path,qc_df=None):
#    from sklearn.svm import OneClassSVM 
#    import importlib.util
#
#    #loni_path = '/home/jagust/adni/pipeline_scripts/csv-maker/output/10-20-2020/FS7_UCBERKELEYAV1451_10-20-2020_regular_tp.csv'
#    if ('pointer' in loni_path):
#        subset_path = '/home/jagust/petcore/qc/scripts/POINTER_QC_Subset_larger.csv'
#    else:
#        subset_path = '/home/jagust/petcore/qc/scripts/ADNI_QC_Subset_larger.csv'
#    loni = pd.read_csv(loni_path)
#    loni['RID'] = loni['RID'].astype(int).astype(str)
#    loni = loni.set_index(['RID','TP'])
#    if qc_df is not None:
#        loni = pd.read_csv(loni_path)
#        loni['RID'] = loni['RID'].astype(int).astype(str)
#        qc_df=qc_df.reset_index(drop=False)
#        loni['PET_Date']=np.nan
#        loni['MRI_Date']=np.nan
#        loni['TP'] = loni['TP'].astype(str)
#        
#        for _ in loni.index:
#            rid=loni.loc[_,'RID']
#            try:
#                tp=loni.loc[_,'TP']
#            except:
#                tp='BL'
#            mdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'MRI_Date']
#            pdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'PET_Date']
#            if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))!=1:
#                if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))==0:
#                    mdate=''
#                    pdate=''
#                else:  
#                    mdate=sorted(list(mdate),reverse=True)[0]
#                    pdate=sorted(list(pdate),reverse=True)[0]
#            else:
#                mdate=list(mdate)[0]
#                pdate=list(pdate)[0]
#            loni.loc[_,'PET_Date']=pdate
#            loni.loc[_,'MRI_Date']=mdate
#
#        loni = loni.set_index(['RID','TP','MRI_Date','PET_Date'])
#
#
#    subset = pd.read_csv(subset_path)
#
#    if qc_df is not None:
#        subset['RID'] = subset['RID'].astype(int).astype(str)
#        qc_df=qc_df.reset_index(drop=False)
#        subset['PET_Date']=np.nan
#        subset['MRI_Date']=np.nan
#        subset['TP'] = ['BL']*len(subset)
#        
#        for _ in subset.index:
#            rid=subset.loc[_,'RID']
#            tp=subset.loc[_,'TP']
#            pdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'PET_Date']
#            mdate=qc_df.loc[(qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid)),'MRI_Date']
#            if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))!=1:
#                if sum(((qc_df.TP.astype(str)==str(tp))&(qc_df.RID.astype(int)==int(rid))))==0:
#                    mdate=''
#                    pdate=''
#                else:  
#                    mdate=sorted(list(mdate),reverse=True)[0]
#                    pdate=sorted(list(pdate),reverse=True)[0]
#            else:
#                mdate=list(mdate)[0]
#                pdate=list(pdate)[0]
#            # print(mdate)
#            subset.loc[_,'PET_Date']=pdate
#            subset.loc[_,'MRI_Date']=mdate
#        subset = subset.set_index(['RID','TP','MRI_Date','PET_Date'])
#    else:
#        subset['TP'] = ['BL']*len(subset)
#        subset.columns = ['RID']+list(subset.columns[1:])
#        subset['RID'] = subset['RID'].astype(int).astype(str)
#        subset=subset.set_index(['RID','TP'])
#    subset = subset.round(decimals=8)
#    
#    spec = importlib.util.spec_from_file_location("searchList", '/home/jagust/tjward/Notebooks/Tyler-Github/Data_Finder/tjw_search_tools.py')
#    st = importlib.util.module_from_spec(spec)
#    spec.loader.exec_module(st)
#    
#    # COLS_OF_INTEREST = (loni.drop(['VISCODE','VISCODE2','EXAMDATE','INJTIME','SCANTIME'],axis=1).dropna(axis=1).columns)
#
#    VOL_COLS = st.searchList(ALL_ROIS,Include=['SIZE'],Exclude=['SUVR','AUTO_FLAG'],return_list=1)[0]
#    SUVR_COLS = st.searchList(ALL_ROIS,Include=[],Exclude=['SIZE','AUTO_FLAG'],return_list=1)[0]
#
#    # SUVR Flag
#    train_group = loni.reindex(subset.index)
#    train_group = train_group[SUVR_COLS]
#    train_group = train_group.dropna()
#    test_group = loni[SUVR_COLS]
#    test_group = test_group.dropna()
#    svm = OneClassSVM(kernel='rbf', nu=0.01, gamma=0.01)
#    svm.fit(train_group)
#    loni.loc[test_group.index,'AUTO_FLAG_SUVR'] = svm.predict(test_group[SUVR_COLS]) 
#
#    # VOL QC
#    train_group = loni.reindex(subset.index)
#    train_group = train_group[VOL_COLS]
#    train_group = train_group.dropna()
#    test_group = loni[VOL_COLS]
#    test_group = test_group.dropna()
#    svm = OneClassSVM(kernel='rbf', nu=0.05, gamma=0.00000000001)
#    svm.fit(train_group)
#    loni.loc[test_group.index,'AUTO_FLAG_VOL'] = svm.predict(test_group[VOL_COLS]) 
#    idx = loni.index
#    loni.index = loni.index.set_levels([idx.levels[0].astype(int), idx.levels[1].astype(str), idx.levels[2].astype(str), idx.levels[3].astype(str)])
#    return (loni['AUTO_FLAG_SUVR'], loni['AUTO_FLAG_VOL'])
