import os,sys,re 
print('*************************************************')
print (sys.version)
# QC sync cluster & google
# Written by: Alice Murphy
from genericpath import exists
from os import popen
from typing import Match, TextIO
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import subprocess
import os.path, time
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from datetime import date as Date
from datetime import timedelta
import csv

from optparse import OptionParser
from googleapiclient import errors
import fnmatch

from automated_flags import *
#from qc_sync_cluster_and_google import *
from google_api_func import *


def glob_re(pattern, path_in):
    '''
    Glob with a regular expression!
    I use this to filter folders inside the data_path directory
    '''
    strings = glob(path_in)
    F0=[]
    for s in strings:
        match = re.search(pattern, s)
        if match:
            F0.append((match.string,match.group(1)))
    return F0

def clean_backup_files(files):
    '''
    Accepts a list of backup file paths
    Example: /home/jagust/petcore/qc/adni/ftp-suvr/backup/

    Keeps most recent 14 files
    Keeps last file for each month
    '''
    from itertools import groupby
    from datetime import datetime


    files = sorted(files, key=lambda x: datetime.strptime(re.search(r'_(\d{2}-\d{2}-\d{4})',x).group(1), '%m-%d-%Y'))
    if len(files) < 14:return
    files_to_keep = files[-14:]

    grouped_year = [list(g) for k, g in groupby(files, lambda x: re.match('.*_\d{2}-\d{2}-(\d{4}).*', x).group(1))]
    for year_list in grouped_year:
        grouped_month = [list(g) for k, g in groupby(year_list, lambda x: re.match('.*_(\d{2})-\d{2}-\d{4}.*', x).group(1))]
        for month in grouped_month:
            '''
            Appends the most recent file for every month to the "files to keep" list.
            '''
            files_to_keep.append(month[-1])

    print('Cleaning backup folder:',os.path.dirname(files[0]))
    for i in files:
        if i not in files_to_keep:
            os.remove(i)
    
    return


    
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



def upload_whoqcing_to_google(pngpathcluster,gauth):
    '''Moves PNG files to visit folders'''
    drive = GoogleDrive(gauth)
    filename=os.path.basename(pngpathcluster)
    folder_link = 'https://drive.google.com/drive/folders/1pFdcsrBGmAvSZ9ggix-STMbwo0amcZSC'
    existing_file=None

    my_v_folder_as_parent_query="'"+folder_link.split('/')[-1]+"' in parents and trashed = false"

    file_list=drive.ListFile({'q': my_v_folder_as_parent_query,'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList()
    for file_ in file_list:
        print('okay')
        file_title=file_['title']
        if file_title==os.path.basename(pngpathcluster):
            print('******Match*********')
            existing_file=file_

    if (existing_file==None)|(existing_file==[]):
        print("Creating...",filename)
        file2upload = drive.CreateFile({'title':filename, 'mimeType':'image/png','parents':[{'id': str(folder_link.split("folders/")[1])}],
                                        'teamDriveId': TEAM_DRIVE_ID})
        file2upload.SetContentFile(pngpathcluster)
    else:
        print("Replacing...",filename)
        # same id will overwrite old file
        file2upload = existing_file
        file2upload.SetContentFile(pngpathcluster)
    file2upload.Upload(param={'supportsAllDrives': True}) # Upload file.
    return


def TSV_on_neurocluster(project_name,tracer_name):
    '''Returns latest QC TSV filepath from neurocluster.'''
    TSVfilenames=glob(f'/home/jagust/petcore/qc/{project_name.lower()}/{tracer_name.lower()}/QC_{project_name.upper()}_{tracer_name.upper()}_*.tsv')
    #reverse = True sorts the lastest dated tsv to first (if neurocluster & google csv have same date, google download is placed first)
    #TSVfilenames.sort(reverse=True)
    TSVfilenames.sort(key=lambda date: datetime.strptime(re.search(r'\d{2}-\d{2}-\d{4}', date).group(), "%m-%d-%Y"),reverse=True)
    if len(TSVfilenames)>0:
        try:
            newestTSVfilename=TSVfilenames[0]
            return newestTSVfilename
        except:
            return 

def upload_TSV(filename,project_name,tracer_name,GAUTH):
    drive = GoogleDrive(GAUTH)
    print(filename,project_name,tracer_name)
    print(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}'))
    folder_id=str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}'))
    filetitle = os.path.splitext(os.path.basename(filename))[0]
    file_ = drive.CreateFile({'title':filetitle,'mimetype':'text/tab-separated-values', 'parents':[{'id': folder_id}],
                                        'teamDriveId': TEAM_DRIVE_ID})
    file_.SetContentFile(filename)
    file_.Upload({'convert': True})

def TSV_on_google(project_name,tracer_name,GAUTH):
    '''Returns sheet object from google.'''
    drive = GoogleDrive(GAUTH)
    folder_link = str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}'))
    QC_title_contains = f'QC_{project_name.upper()}_{tracer_name.upper()}'
    file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and title contains '{QC_title_contains}'"}).GetList()
    TSVfilenames=[]
    TSVs=[]
    for file_ in file_list:
        if (project_name.upper() in file_['title']) and (tracer_name.upper() in file_['title']) and ("tsv" not in file_['title']):
            TSVfilenames.append(file_['title'])
            TSVs.append(file_)
    if len(TSVfilenames) == 0:
        print('No TSV on Google')
        return None
    TSVfilenames.sort(key=lambda date: datetime.strptime(re.search(r'\d{2}-\d{2}-\d{4}', date).group(), "%m-%d-%Y"),reverse=True) 
    print(TSVfilenames)
    newestTSVfilename=TSVfilenames[0]
    print( [i['title'] for i in TSVs] )
    newestTSV=[TSVs[x] for x in range(len(TSVs)) if TSVs[x]['title'] == newestTSVfilename]
    print( [i['title'] for i in TSVs] )
   # print(len(newestTSV))
    if len(newestTSV)>=1:
        return newestTSV[0]
    else:
        return None

def fetchPNGs_fromGoogle(project_name,tracer_name,visit,GAUTH):
    '''Returns:
            PNG title list
            PNG date modified list
            PNG file objects
    '''
    drive = GoogleDrive(GAUTH)
    file_title_list=[]
    file_dateModified_list=[]
    actual_files_from_google=[]

    visit_folder_link = str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}/{visit}'))
    tracer_folder_link = str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}'))

    my_v_folder_as_parent_query="'"+visit_folder_link+"' in parents and trashed = false"

    for file in drive.ListFile({'q': my_v_folder_as_parent_query,'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList():
        file_title=file['title']
        if visit_folder_link == file['parents'][0]['id']:
            if (tracer_name.upper() in file_title):
                date_str=file['modifiedDate'].replace('T',' ')
                date_str=date_str.replace('Z',' ')[:-5]
                file_title_list.append(file['title'])
                file_dateModified_list.append(datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S') - timedelta(hours = 7))
                actual_files_from_google.append(file)

    return file_title_list, file_dateModified_list, actual_files_from_google



def getNeedsQC_linkList(project_name,tracer_name,GAUTH):
    '''Returns:
            PNG title list
            PNG date modified list
            PNG file objects
    '''
    drive = GoogleDrive(GAUTH)
    file_title_list=[]
    file_dateModified_list=[]
    actual_files_from_google=[]

    needsqc_folder_link =  str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}/needs_qc'))
    tracer_folder_link =  str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}'))
    
    my_needsqc_folder_as_parent_query=f"'{tracer_folder_link}' in parents and title contains 'needs_qc' and hidden = false and trashed = false"

    for folder in drive.ListFile({'q': my_needsqc_folder_as_parent_query,'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList():
        if (needsqc_folder_link == folder['alternateLink'].split('/')[-1]) or (needsqc_folder_link == folder['selfLink'].split('/')[-1]):
            if tracer_folder_link in folder['parents'][0]['id']:
                my_needsqc_files_q="mimeType = 'application/vnd.google-apps.shortcut' and trashed = false and hidden = false"
                for file_ in drive.ListFile({'q': my_needsqc_files_q,'supportsAllDrives': True, 'includeItemsFromAllDrives': True}).GetList():
                    if needsqc_folder_link in file_['parents'][0]['id']:
                        date_str=file_['modifiedDate'].replace('T',' ')
                        date_str=date_str.replace('Z',' ')[:-5]
                        file_title_list.append(file_['title'])
                        file_dateModified_list.append(datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S'))
                        actual_files_from_google.append(file_)
    # print("total # files in needs qc = ",len(file_title_list))
    return file_title_list, file_dateModified_list, actual_files_from_google

def take_down_spreadsheets(project_name,tracer_name,GAUTH=''):
    '''Take down spreadsheet. Used while creating a new spreadsheet / QC images on the neurocluster so that no visual reads are overwritten.'''
    drive = GoogleDrive(GAUTH)
    file_list = drive.ListFile({'q': "trashed = false and title contains 'QC_'"}).GetList()
    #filenames1 = [i['title'] for i in file_list]
    #print(filenames1)
    TSVfilenames=[]
    TSVs=[]
    for file_ in file_list:
        if (project_name.upper() in file_['title']) and (tracer_name.upper() in file_['title']):  #and ("tsv" not in file_['title'])
            TSVfilenames.append(file_['title'])
            TSVs.append(file_)
    for file_ in TSVs:
        print('Trying to delete:',file_['title'])
        try:file_.Delete()
        except:print('FAIL! Can not delete file',file_['title'])

def create_new_Google_TSV(project_name,tracer_name,GAUTH=''):
    neuroclusterPath=TSV_on_neurocluster(project_name,tracer_name)
    googleSheet=TSV_on_google(project_name,tracer_name,GAUTH)
    if googleSheet != None:
        print('A QC CSV already exists here -',googleSheet['title'])
        return
    upload_TSV(neuroclusterPath,project_name,tracer_name,GAUTH)
    googleSheet=TSV_on_google(project_name,tracer_name,GAUTH)
    if googleSheet != None:
        print('Success!')
    else:
        print('Failed!')

def compare_QC_review_files(project_name,tracer_name,GAUTH='',upordown=""):
    '''
    Compares the cluster file creation time & google sheet modification time (down to the second)
        During uploads --> If the cluster date is LATER than the google date, the sheet will upload to google and update the file's date to today.
        During downloads --> If the file on google is modified LATER than the cluster file creation, the google sheet will download with "_temp_" and today's date.
    '''
    print("")
    print("")
    print("--------------------------------------")
    print(project_name.upper(),tracer_name.upper(),today_date)
    print("--------------------------------------")

    #find newest neurocluster csv
    neuroclusterPath=TSV_on_neurocluster(project_name,tracer_name)

    #find newest google sheet
    googleSheet=TSV_on_google(project_name,tracer_name,GAUTH)
    if googleSheet == None:
        print('No TSV found on Google')
        return

    print('Newest TSV on Google:', googleSheet['title'])

    date_str=str(googleSheet['modifiedDate'])
    match = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})',date_str)
    date_str='{} {}'.format(match.group(1), match.group(2))
    date_google = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

    print(neuroclusterPath,googleSheet['title'])

    if neuroclusterPath !="":
        # date_cluster is the modify date according to the server
        date_cluster=datetime.strptime(datetime.fromtimestamp(os.path.getmtime(neuroclusterPath)).strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')
        if (upordown=='down'): #and (date_cluster<date_google):
            print("Found updated QC file on google")
            #create filename for google data
            #googleNeuroclusterPath=neuroclusterPath.replace(neuroclusterPath.split(tracer_name.upper()+'_')[1],"temp_"+today_date_str)+".tsv"
            new_google_tsv_path = f'QC_{project_name.upper()}_{tracer_name.upper()}_{today_date_str}.tsv'
            new_google_tsv_title = f'QC_{project_name.upper()}_{tracer_name.upper()}_{today_date_str}'
            googleNeuroclusterPath = os.path.join(os.path.dirname(neuroclusterPath), \
                    new_google_tsv_path)
            # Download googleSheet CSV from google to temp_
            googleSheet.GetContentFile(googleNeuroclusterPath, mimetype='text/tab-separated-values')
            #raise ValueError('TEST')
            move_tsv_header_to_first_line(googleNeuroclusterPath,QC_index_cols)
            googleDF=pd.read_csv(googleNeuroclusterPath,low_memory=False,dtype=str,sep='\t',header=0)
            #googleDF = googleDF.loc[~(googleDF['ID'] == 'ID')]


            #neuroclusterDF=pd.read_csv(neuroclusterPath,low_memory=False,dtype=str,sep='\t')
            #if False:#ctemp bool(neuroclusterDF.equals(googleDF)):
            #if bool(neuroclusterDF.equals(googleDF)):
            #    print("----> Removing temp google file.")
            #    subprocess.run(['rm',googleNeuroclusterPath])
            #    mostrecent_path=neuroclusterPath
            #else:
            #NeuroclusterPath = googleNeuroclusterPath.replace('_temp_','_')
            #print("Renaming temp google file ---->",NeuroclusterPath)
            # Overwriting neurocluster data with Google data
            
            #subprocess.run(['mv',googleNeuroclusterPath,NeuroclusterPath])
            A,B = os.path.split(googleNeuroclusterPath)
            B = B.replace('.tsv','.tar.gz')
            backup_path = os.path.join(A,'backup','backup_'+B)
            if not os.path.exists(os.path.dirname(backup_path)):
                os.makedirs(os.path.dirname(backup_path))
            i=2
            #while os.path.exists(backup_path):
            #    backup_path = os.path.join(A,'backup',f'backup{i}_'+B)
            #    i+=1

            all_backup_files = glob(os.path.join(A,'backup','backup_*.tar.gz'))
            clean_backup_files(all_backup_files)
            all_csv_files = glob(os.path.join(os.path.dirname(neuroclusterPath), \
                                                    f'QC_{project_name.upper()}_{tracer_name.upper()}_??-??-????.tsv'))
            clean_backup_files(all_csv_files)

            # Tar it for archiving
            subprocess.run(['tar','-czvf',backup_path,googleNeuroclusterPath])

            #NEW_nameForGoogle="QC_"+str(project_name.upper())+"_"+tracer_name.upper()+"_"+today_date_str
            googleSheet['title']=new_google_tsv_title      

            googleSheet.Upload()
            mostrecent_path=googleNeuroclusterPath
            # Create study lead DF with only flagged rows
            flagged_df = googleDF.copy()
            flagged_df = flagged_df.loc[(~np.isin(flagged_df['Usability'],['','1',1,np.nan,'1.0'])) & ~(flagged_df['Usability'].isna())]
            flagged_df_path = os.path.join(os.path.dirname(googleNeuroclusterPath), 'FLAGGED_'+os.path.basename(googleNeuroclusterPath))
            print('Saving flagged df -',flagged_df_path)
            flagged_df.to_csv(flagged_df_path,sep='\t',index=False, quoting=csv.QUOTE_ALL)

            all_flagged_csv_files = glob(os.path.join(os.path.dirname(neuroclusterPath), \
                                                    f'FLAGGED_QC_{project_name.upper()}_{tracer_name.upper()}_??-??-????.tsv'))
            clean_backup_files(all_flagged_csv_files)
                

            return  mostrecent_path
        elif (upordown=='up'): #and (date_cluster>date_google):
            print("Found updated QC file on cluster")
            #create filename for google data
            NEW_nameForGoogle="QC_"+str(project_name.upper())+"_"+tracer_name.upper()+"_"+today_date_str
            print("uploading ",NEW_nameForGoogle)      
            #NewNeuroclusterPath=neuroclusterPath.split(tracer_name.upper()+'_')[0]+today_date_str+'.tsv'
            #NewNeuroclusterPath="QC_"+str(project_name.upper())+"_"+tracer_name.upper()+"_"+today_date_str+'.tsv'
            #print('Reading: ',neuroclusterPath)
            #neuroclusterDF=pd.read_csv(neuroclusterPath,low_memory=False,dtype=str,sep='\t')
            #print('Saving:', NewNeuroclusterPath)
            #neuroclusterDF.to_csv(NewNeuroclusterPath,index=False,sep='\t')
            mostrecent_path=neuroclusterPath
            googleSheet.SetContentFile(neuroclusterPath)
            googleSheet['title']=NEW_nameForGoogle        
            googleSheet.Upload()
            return mostrecent_path
        else:
            mostrecent_path=neuroclusterPath
            return mostrecent_path
    else:
        print("-----------> NO EXISTING QC FILE ON CLUSTER -------------> Attempting google download.")
        try:
            googleNeuroclusterPath="/home/jagust/petcore/qc/"+project_name.lower()+"/"+tracer_name.lower()+"/QC_"+project_name.upper()+"_"+tracer_name.upper()+'_temp_'+today_date_str+".tsv"
            googleSheet.GetContentFile(googleNeuroclusterPath, mimetype='text/tab-separated-values')
            print('Google sheet --> ',googleNeuroclusterPath)
            print('Neurocluster --> ',"None")
            return googleNeuroclusterPath
        except:
            return 

def QCspreadsheet_sync(project_name,tracer_name,GAUTH,upordown=""):
    '''Makes sure most-up-to-date file is on drive and cluster
        Prints most recent file paths'''
    filenames=[]
    qcfile_on_cluster = compare_QC_review_files(project_name,tracer_name,GAUTH,upordown)
    print("\n\nMost recent -> ",project_name,tracer_name,qcfile_on_cluster)
    #elif project_name=='adni':
    #    for tracer_name in ['fbp','fbb','ftp']:
    #        # Find new or modified QC images on cluster and add to Google
    #        qcfile_on_cluster=compare_QC_review_files(project_name,tracer_name,GAUTH,upordown)
    #        filenames.append(qcfile_on_cluster) 
    #        print("\n\nMost recent -> ",project_name,tracer_name,qcfile_on_cluster)     
    return 

def upload_png_to_google(pngpathcluster,project_name,tracer_name,existing_file,gauth):
    '''Moves PNG files to visit folders'''
    drive = GoogleDrive(gauth)
    filename=os.path.split(pngpathcluster)[1]
    if '/v1/' in pngpathcluster:
        folder_name='v1'
    elif '/v2/' in pngpathcluster:
        folder_name='v2'
    elif '/v3/' in pngpathcluster:
        folder_name='v3'
    elif '/v4/' in pngpathcluster:
        folder_name='v4'
    elif '/v5/' in pngpathcluster:
        folder_name='v5'
    elif '/v6/' in pngpathcluster:
        folder_name='v6'
    else:
        raise ValueError("Error: timepoints greater than v6 not yet set-up")

    folder_link =  str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}/{folder_name}'))
    if (existing_file==None)|(existing_file==[]):
        print("Creating...",filename)
        file2upload = drive.CreateFile({'title':filename, 'mimeType':'image/png','parents':[{'id': str(folder_link)}], 
                                        'teamDriveId': TEAM_DRIVE_ID})
        file2upload.SetContentFile(pngpathcluster)
    else:
        print("Replacing...",filename)
        # same id will overwrite old file
        file2upload = existing_file
        file2upload.SetContentFile(pngpathcluster)
    file2upload.Upload(param={'supportsTeamDrives': True}) # Upload file.
    return 



def newQCimages_toGoogle(project_name,tracer_name,GAUTH):
    '''
    Loops through QC PNGs and looks for title match on Google.
        If found, checks when the 'cluster PNG' and 'Google PNG' were last modified.
            If Cluster PNG is newer than the Google PNG, the old Google PNG is modified with the new image content.
            If Cluster PNG is older than the Google PNG, nothing happens.
        If no match on Google, new PNG is uploaded from Cluster to Google.
        If on google and not cluster, delete from google.
    Output file saves modifications to /home/jagust/qc/petcore/scripts/transfer_history/
    '''
    new_images=pd.DataFrame(columns=['QC_images','Modifications'])
    n_newscans=0
    n_replacedscans=0
    edittedscans=[]
    mods=[]
    visits = glob_re(r'\/(v\d{1,3})\/',f'/home/jagust/petcore/qc/{project_name}/{tracer_name}/v*/')
    for vpath,visit in visits:
        PNGcluster=glob(os.path.join(vpath,'*.png'))
        # Sort images by filename
        PNGcluster.sort(key=lambda x: os.path.basename(x))
        
        PNGtitles_drive,DATESdrive,DRIVEfiles=fetchPNGs_fromGoogle(project_name,tracer_name,visit,GAUTH)
        orphaned_GoogleImages = list(DRIVEfiles.copy())
        for _ in range(len(PNGcluster)):
            pngpathcluster=PNGcluster[_]
            datecluster=datetime.strptime(datetime.fromtimestamp(os.path.getmtime(pngpathcluster)).strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')
            filename=os.path.split(pngpathcluster)[1]
            # For every PNG on Google, search for a match to the file on Neurocluster.
            DRIVEbool=[bool(_==filename) for _ in PNGtitles_drive]
            drivefile=[i for (i,v) in zip(DRIVEfiles, DRIVEbool) if v]
            drivedate=[i for (i,v) in zip(DATESdrive, DRIVEbool) if v]

            for file_ in drivefile:
                # The image is on Google and on the cluster
                orphaned_GoogleImages.remove(file_)

            if sum(DRIVEbool)>1:
                # Multiple PNGs on Cluster with same name?
                for d in drivefile:
                    d.Delete()
                upload_png_to_google(pngpathcluster,project_name,tracer_name,existing_file=None,gauth=GAUTH)
                new_images.loc[len(new_images)] = [filename, 'replaced']
                n_replacedscans=n_replacedscans+1
            elif sum(DRIVEbool)==0:
                # NEW images
                new_images.loc[len(new_images)] = [filename, 'new']
                n_newscans=n_newscans+1
                upload_png_to_google(pngpathcluster,project_name,tracer_name,existing_file=drivefile,gauth=GAUTH)
            elif sum(DRIVEbool)==1:
                ## CHECK GOOGLE FILE UPLOAD DATE AND CLUSTER FILE CREATION DATE
                drivedate=drivedate[0]
                drivefile=drivefile[0]
                if datecluster>drivedate:
                    # REPLACE images
                    new_images.loc[len(new_images)] = [filename, 'replaced']
                    n_replacedscans=n_replacedscans+1
                    print('uploading ',pngpathcluster)
                    upload_png_to_google(pngpathcluster,project_name,tracer_name,existing_file=drivefile,gauth=GAUTH)

        for file_ in orphaned_GoogleImages:
            print('Removing orphan -',file_['title'])
            file_.Delete()

      #  new_images['QC_images']=edittedscans
      #  new_images['Modifications']=mods
        if len(edittedscans)>0:
            today_date = Date.today()
            new_images.to_csv('transfer_history/'+project_name+'_'+tracer_name+'_transfer_'+str(today_date) +'.tsv',index=False,sep='\t', quoting=csv.QUOTE_ALL)
        print(project_name.upper(),tracer_name.upper(),"--->",n_newscans," new & ",n_replacedscans," replaced scans / images")
        
def createPNG_link(project_name,tracer_name,filename,GAUTH):
    rid_=filename.split('_')[0]
    tp_=filename.split('_')[1]
    #tracer_=filename.split('_')[2].replace('.png','').lower()
    PNGtitles_drive,DATESdrive,DRIVEfiles=fetchPNGs_fromGoogle(project_name,tracer_name,tp_,GAUTH)
    if filename in PNGtitles_drive:
        folder_link =  str(get_qc_folder_id(GAUTH,f'{project_name}/{tracer_name}/needs_qc'))
        drive = GoogleDrive(GAUTH)
        ix=[_==filename for _ in PNGtitles_drive]
        if sum(ix)==1:
            file_g=[DRIVEfiles[idx] for idx in range(len(DRIVEfiles)) if ix[idx]==True][0]
            shortcut_metadata = {
                        'name': filename,
                        'mimeType': 'application/vnd.google-apps.shortcut',
                        'parents':[{'id': folder_link}], 
                        'teamDriveId': TEAM_DRIVE_ID,
                        'shortcutDetails': {'targetId': file_g.get('id')}}
            newlink=drive.CreateFile(shortcut_metadata)
            newlink.Upload(param={'supportsTeamDrives': True}) # Upload file.

def refreshNeedsQC(project_name,tracer_name,GAUTH):
    drive_tmp = GoogleDrive(GAUTH)
    visits = glob_re(r'\/(v\d{1,3})\/',f'/home/jagust/petcore/qc/{project_name}/{tracer_name}/v*/')
    visit_codes = [i[1] for i in visits]
    n_visits = len(visits)
    googleSheet=TSV_on_google(project_name,tracer_name,GAUTH)
    googleNeuroclusterPath=f'/home/jagust/petcore/qc/{project_name.lower()}/{tracer_name.lower()}/QC_{project_name.upper()}_{tracer_name.upper()}_temp_{today_date_str}.tsv'
    googleSheet.GetContentFile(googleNeuroclusterPath, mimetype='text/tab-separated-values')
    move_tsv_header_to_first_line(googleNeuroclusterPath,QC_index_cols)
    googleDF=pd.read_csv(googleNeuroclusterPath,low_memory=False,dtype=str,sep='\t',header=0)
    #googleDF = googleDF.loc[~(googleDF['ID'] == 'ID')]
    # Drop the rows with replaced MRIs
    googleDF=googleDF[googleDF["TP"].str.contains("__")==False]
    googleDF = googleDF.set_index(['ID','TP'])
    QCed = list(googleDF.loc[(googleDF['Review Date'].notnull())&(googleDF['Review Date'] != "NULL")].index.values)
    #QCedRID=[str(_) for _ in googleDF.loc[(googleDF['Review Date'].notnull())&(googleDF['Review Date'].="NULL"),'ID'].astype(int)]
    #QCedRID=googleDF.loc[(googleDF['Review Date'].notnull())&(googleDF['Review Date'].="NULL"),'ID'].astype(str)
    #QCedTP=googleDF.loc[(googleDF['Review Date'].notnull())&(googleDF['Review Date'].="NULL"),'TP'].astype(str)
    #QCed=zip(QCedRID,QCedTP) 
    print('*************************************************************')
    removed_list=[]

    not_QCed = list(googleDF.loc[(~googleDF['Review Date'].notnull())|(googleDF['Review Date']=="NULL")].index.values)
    #not_qcedRID=[str(_) for _ in googleDF.loc[(~googleDF['Review Date'].notnull())|(googleDF['Review Date'].="NULL"),'ID'].astype(str)]
    #not_qcedTP=[str(_) for _ in googleDF.loc[(~googleDF['Review Date'].notnull())|(googleDF['Review Date'].="NULL"),'TP'].astype(str)]
    #print('NOT QCd List length:',len(not_qcedRID))
    ##not_qcedTP=googleDF.loc[~googleDF['Review Date'].notnull(),'TP'].astype(str)
    #not_qced=zip(not_qcedRID,not_qcedTP)
    added_list=[]

    print('             total QCed: n=',len(QCed))
    print('             total not QCed: n=',len(not_QCed))

    # remove completed review links
    for cred_i in [0,]:
        #
        # This for loop is used to delete old or broken QC link folder. Sometimes the QC links are under different authentication
        # Thus, I loop through several different credentials and delete old or broken QC links. 
        # Never use anything but default credentials (alice_creds.txt) for creating links or folders. 
        #
        #GAUTH_onlydelete, CREDENTIALS_onlydelete = initiate_auth(cred_i)
        needsqc_titlelist, needsqc_datelist, needsqc_links = getNeedsQC_linkList(project_name,tracer_name,GAUTH)
        for i in range(len(needsqc_links)):
            title_=needsqc_titlelist[i]
            rid_=title_.split('_')[0]
            tp_=title_.split('_')[1]
            item_=needsqc_links[i]
            item_.FetchMetadata(fields='shortcutDetails')
            link_file_id = item_.get('shortcutDetails')['targetId']
            gfile_link = drive_tmp.CreateFile({'id': link_file_id})
            try:
                gfile_link.FetchMetadata()
            except:
                gfile_link == dict()
            if 'title' not in dict(gfile_link.items()):
                try:
                    print('Deleting dead link (title empty)', rid_,tp_)
                    item_.Delete()
                except Exception as e: 
                    print('Could not delete broken link with cred',cred_i,item_['title'])
                    print(' - moved to Backup/old_links')
                    item_['parents'] = [{"kind": "drive#parentReference", "id": '1LlDiUo3lqoDIYU5L4VeZRIgkbvob6mlB'}]
                    item_.Upload()
                    #breakpoint()

                continue
            #else:
            #    print('Shortcut Details:',item_.get('shortcutDetails'))
            if (rid_, tp_) in QCed:
                print('Deleting Needs QC Link (has been QCd):',title_)
                try:
                    item_.Delete()
                except Exception as e: 
                    item_['parents'] = [{"kind": "drive#parentReference", "id": '1LlDiUo3lqoDIYU5L4VeZRIgkbvob6mlB'}]
                    item_.Upload()
                    print('Could not delete old link - moved to Backup/old_links')

    needsqc_titlelist, needsqc_datelist, needsqc_links = getNeedsQC_linkList(project_name,tracer_name,GAUTH)

    # check for existing links and create when necessary
    google_titlelist=[]
    google_datelist=[]
    google_pnglist=[]
    for tp_ in visit_codes:
        titlelist, datelist, googlepngs = fetchPNGs_fromGoogle(project_name,tracer_name,tp_,GAUTH)
        google_pnglist=google_pnglist+googlepngs
        google_datelist=google_datelist+datelist
        google_titlelist=google_titlelist+titlelist
    for rid, tp in not_QCed:
        filter_str = f'{rid}_{tp}_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_{tracer_name.upper()}.png'
        filename = fnmatch.filter(google_titlelist,filter_str   )
        if len(filename) == 0:
            print(f'Could not find QC image! - {filter_str}')
            continue
        filename = filename[0]
        if filename not in list(needsqc_titlelist):
            #print(f'{filename} not in list(needsqc_titlelist)')
            if filename in list(google_titlelist):
                print('Linking',rid,tracer_name,tp)
                createPNG_link(project_name,tracer_name,filename,GAUTH)
            #else:
            #    print(f'{filename} in list(google_titlelist)')
        #else:
        #    print(f'{filename} in list(needsqc_titlelist)')
    print("----> Removing temp google file.")
    subprocess.run(['rm',googleNeuroclusterPath])
    return

def modifyBusySign(project,tracer,GAUTH,direction):
    '''A sign will show up on Google if the QC is processing, explaining why there is no TSV. up or down.'''
    print('Running modifyBusySign -',project,tracer,'GAUTH',direction)

    drive = GoogleDrive(GAUTH)
    folder_link =  str(get_qc_folder_id(GAUTH,f'{project}/{tracer}'))
    localBusySign = os.path.join(f'/home/jagust/petcore/qc/scripts/required/BusySign_{project}_{tracer}.txt')
    GoogleBusyFile = os.path.join(f'/home/jagust/petcore/qc/scripts/required/BusySign.txt')

    file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and Title='BusySign.txt'"}).GetList()

    if direction == 'down':
        for file_ in file_list:
            print('Trying to delete:',file_['title'])
            try:file_.Delete()
            except:print('FAIL! Can not delete file',file_['title'])
        if os.path.exists(localBusySign):
            print('Deleting -',localBusySign)
            os.remove(localBusySign)
        else:
            print('No local busy sign -',localBusySign)

    if direction == 'up':
        if len(file_list) > 0:
            return
        else:
            file_ = drive.CreateFile({'title':os.path.basename(GoogleBusyFile),'mimetype':'text/plain', 'parents':[{'id': str(folder_link)}],
                                        'teamDriveId': TEAM_DRIVE_ID})
            file_.SetContentFile(GoogleBusyFile)
            file_.Upload({'convert': False})
        with open(localBusySign, 'w') as out_:
            out_.write('QC code busy.')


def create_google_folder(project,tracer,GAUTH):
    '''
    Only used to initiate a new project. Can be easily modified to just add tracers or just add timepoints.

    Must first create the parent folder and add the link to the link dictionary

     python qc_sync_cluster_and_google.py -p "scan" -t "fbp-suvr" --createFolders

    '''
    print('Project folders:')
    drive = GoogleDrive(GAUTH)
    #folder_link=QC_FOLDER # QC google folder for creating a new project folder.
    #folder_link='1VQqfY7RKJ0_w4G88zJrTyBHLPs-E6agp'
    folder_link =  str(get_qc_folder_id(GAUTH,f'{project}/{tracer}'))
    project_title = project.lower()
#    file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and Title='{project_title}'"}).GetList()
#    if len(file_list) > 0:
#        for i in file_list:
#            print('Found -',i['id'])
#    
#    print('parent folder:',folder_link)
#
#    if len(file_list) == 0:
#        file_metadata = {
#                      'title': project_title,
#                      'parents': [{"kind": "drive#parentReference", "id": folder_link}],
#                      'supportsAllDrives': 'true',
#                      'shared':'true',
#                      'mimeType': 'application/vnd.google-apps.folder'
#                    }
#        pfolder = drive.CreateFile(file_metadata)
#        pfolder.Upload(param={'supportsAllDrives': True})
#        print(project,pfolder['id'])
#        folder_link = pfolder['id']
#    file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and Title='{tracer}'"}).GetList()
#    if len(file_list) == 0:
#        file_metadata = {
#                      'title': tracer,
#                      'parents': [{"kind": "drive#parentReference", "id": folder_link}],
#                      'supportsAllDrives': 'true',
#                      'shared':'true',
#                      'mimeType': 'application/vnd.google-apps.folder'
#                    }
#        pfolder = drive.CreateFile(file_metadata)
#        pfolder.Upload(param={'supportsAllDrives': True})
#        print(tracer,pfolder['id'])
#    folder_link = pfolder['id']
    for i in range(1, 9):
        folder_tp = f'v{str(i)}'
        file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and Title='{folder_tp}'"}).GetList()
        if len(file_list) == 0:
            file_metadata = {
                          'title': folder_tp,
                          'parents': [{"kind": "drive#parentReference", "id": folder_link}],
                          'supportsAllDrives': 'true',
                          'shared':'true',
                          'mimeType': 'application/vnd.google-apps.folder'
                        }
            pfolder = drive.CreateFile(file_metadata)
            pfolder.Upload(param={'supportsAllDrives': True})
            print(folder_tp,pfolder['id'])
    
    folder_tp = f'needs_qc'
    file_list = drive.ListFile({'q': f"'{folder_link}' in parents and trashed=false and Title='{folder_tp}'"}).GetList()
    if len(file_list) == 0:
        file_metadata = {
                      'title': folder_tp,
                      'parents': [{"kind": "drive#parentReference", "id": folder_link}],
                      'supportsAllDrives': 'true',
                      'shared':'true',
                      'mimeType': 'application/vnd.google-apps.folder'
                    }
        pfolder = drive.CreateFile(file_metadata)
        pfolder.Upload(param={'supportsAllDrives': True})
        print(folder_tp,pfolder['id'])

    print('Done.')


def QCisBusy(project,tracer):
    localBusySign = os.path.join(f'/home/jagust/petcore/qc/scripts/required/BusySign_{project}_{tracer}.txt')
    if os.path.exists(localBusySign):
        return True
    else:
        return False


if __name__ == "__main__":
    opt_parser = OptionParser()
    opt_parser.add_option("-p", "--project", dest="project", default='', help='enter adni,pointer (script only set up for pointer)')
    opt_parser.add_option("-t", "--tracer", dest="tracer", default='', help='tracer-proc: ftp-dvr, fbb-suvr, etc')
    opt_parser.add_option("--updateBusy", dest="updateBusy", default='', help='upload or take down busy sign')
    opt_parser.add_option("-r", "--downloadreviews", dest="downloadreviews", action="store_true", default=False, help="download reviews from google to neurocluster")
    opt_parser.add_option("-R", "--uploadrows", dest="uploadrows", action="store_true", default=False, help="download reviews from google to neurocluster")
    opt_parser.add_option("-I", "--uploadPNGs", dest="uploadPNGs", action="store_true", default=False, help="upload PNGs from neurocluster to google")
    opt_parser.add_option("-Q", "--needsQC", dest="needsQC", action="store_true", default=False, help="link PNGs that need review")
    opt_parser.add_option("--delete", dest="delete", action="store_true", default=False, help="Take down spreadsheet from Google")
    opt_parser.add_option("--newTSV", dest="newTSV", action="store_true", default=False, help="Upload a new TSV to google")
    opt_parser.add_option("--autoPlots", dest="autoPlots", action="store_true", default=False, help="Automated Plots")
    opt_parser.add_option("--createFolders", dest="createFolders", action="store_true", default=False, help="Create google folders for a new project tracer ")
    opt_parser.add_option("--ignoreBusy", dest="ignoreBusy", action="store_true", default=False, help="Ignore the busy sign ")
    opt_parser.add_option("--mrifree", dest="mrifree", action="store_true", default=False, help="Run for MRIfree")
    (options, args) = opt_parser.parse_args()
    project = options.project.lower()
    project_name = options.project.lower()
    tracer = options.tracer.lower()
    mrifree = options.mrifree

    today_date = Date.today()
    today_date_str=today_date.strftime("%m-%d-%Y")
    NEW_PATH=f'/home/jagust/petcore/qc/{project}/{tracer}/QC_{project.upper()}_{tracer.upper()}_{today_date_str}.tsv'
    NEW_NAME=f'QC_{project.upper()}_{tracer.upper()}_{today_date_str}.tsv'

    send_reviews_down = options.downloadreviews
    send_rows_up = options.uploadrows
    send_png_up = options.uploadPNGs
    needs_qc = options.needsQC
    autoPlots = options.autoPlots
    delete = options.delete
    newTSV = options.newTSV
    updateBusy = options.updateBusy
    createFolders = options.createFolders
    feedback = 'Nothing done'

    if mrifree:
        QC_columns = ['ID','TP','PET Date','Image Notes','QC Notes','Reviewer Initials','Review Date', 'Usability','Planned Intervention','RA Initials','Reviewed QC Flag','PET Cerebellum Overlap', 'PET Cerebrum Overlap']
        QC_columns_noindex = QC_columns[3:]
        QC_index_cols = ['ID','TP','PET Date']
    else:
        QC_columns = ['ID','TP','PET Date','MRI Date','Image Notes','QC Notes','Reviewer Initials','Review Date', 'Usability','Planned Intervention','RA Initials','Reviewed QC Flag','L/R Symmetry','PET Cerebellum Overlap', 'PET Cerebrum Overlap','Longitudinal MRI Alignment']
        QC_columns_noindex = QC_columns[4:]
        QC_index_cols = ['ID','TP','PET Date','MRI Date']


    try:
        GAUTH, CREDENTIALS = initiate_auth()

        # This can occur even if images are being made in the background
        if send_png_up:
            newQCimages_toGoogle(project,tracer,GAUTH)
            feedback="Complete: if new QC images, added to Google."

        # Check if images / new QC TSV is being created in the background
        if QCisBusy(project,tracer) and (options.ignoreBusy == False):
            feedback = 'SGE is processing new QC images. Waiting. Careful using the flag "--ignoreBusy"'
            print(feedback)
            sys.exit()

        # Nothing below this will run if QC images are being created in background
        if updateBusy == 'up':
            modifyBusySign(project,tracer,GAUTH,'up')
        if updateBusy == 'down':
            modifyBusySign(project,tracer,GAUTH,'down')
        if send_reviews_down:
            QCspreadsheet_sync(project,tracer,GAUTH,'down')
            feedback="Complete: if new QC reviews, copied down to cluster."
        if send_rows_up:
            QCspreadsheet_sync(project,tracer,GAUTH,'up')
            feedback="Complete: if new QC rows, added to Google sheet."
        if needs_qc:
            refreshNeedsQC(project,tracer,GAUTH)
            feedback="Complete: if new QC reviews, refresh needs_qc."
        if autoPlots:
            upload_whoqcing_to_google('../images/Who_is_QCing.png',GAUTH)
            upload_whoqcing_to_google('../images/Who_is_QCing_plain.png',GAUTH)
            feedback='Complete: Who is QCing'
        if delete:
            take_down_spreadsheets(project,tracer,GAUTH)
            feedback='Complete: Take down spreadsheet'
        if newTSV:
            create_new_Google_TSV(project,tracer,GAUTH)
        if createFolders:
            create_google_folder(project,tracer,GAUTH)
    except Exception as e:
        print(e)
        raise Exception("Specify valid destination type: -p <adni,pointer> and valid flags (-r and -R for upload and download)")
    print(feedback)

