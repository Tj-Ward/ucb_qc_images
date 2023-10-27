#!/usr/bin/env python

print('Starting python')
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
#from IPython.display import HTML
from datetime import datetime
from datetime import timedelta
import os,re,glob
import subprocess


QC_columns = ['ID','TP','PET Date','MRI Date','Image Notes','QC Notes','Reviewer Initials','Review Date', 'Usability','Planned Intervention','RA Initials','Reviewed QC Flag','L/R Symmetry','PET Cerebellum Overlap', 'PET Cerebrum Overlap','Longitudinal MRI Alignment']
QC_columns = ['ID','TP','PET Date','MRI Date','Reviewer Initials','Review Date','Usability']
QC_columns_noindex = QC_columns[4:]

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

squid_png = '/home/jagust/petcore/qc/scripts/required/squid.png'
squid_img = Image.open(squid_png)

ocean_png = '/home/jagust/petcore/qc/scripts/required/ocean.png'
ocean_img = Image.open(ocean_png)

finish_png = '/home/jagust/petcore/qc/scripts/required/finish_line.png'
finish_img = Image.open(finish_png)

ACTIVE_PROJECTS = [\
            ['bacs','pib-dvr'],\
            ['bacs','ftp-suvr'],\
            #['bacs','pib-suvr'],\
            ['adni','ftp-suvr'],\
            ['adni','fbb-suvr'],\
            ['adni','fbp-suvr'],\
            #['ucsf','pib-suvr'],\
            ['ucsf','pib-dvr'],\
            ['ucsf','ftp-suvr'],\
            ['pointer','mk6240-suvr'],\
            ['pointer','fbb-suvr'],\
            ['head','ftp-suvr'],\
            ['head','mk6240-suvr'],\
            ['head','pib-suvr'],\
            ['head','fbb-suvr'],\
            ['head','nav-suvr'],\
           ]

names = {'TJW':'Tyler Ward',\
        'AEM':'Alice Murphy',\
        'JQL':'JiaQie Lee',\
        'TC':'Trevor Chadwick',\
        'JL':'Ji Yeon Lee',\
        'CT':'Cheyenne Tsai',\
        'WPT':'Wesley Thomas',\
        'MB':'Marisa Becerra',\
        'KC':'Kaitlin Cassady',\
        'XC':'Xi Chen',\
        'CSF':'Corrina Fonseca',\
        'CF':'Corrina Fonseca',\
        'JG':'Joe Giorgio',\
        'FH':'Feng Han',\
        'ML':'Molly Lapoint',\
        'SM':'Samira Maboudian',\
        'SP':'Stefania Pezzoli',\
        'TT':'Tyler Toueg',\
        'TNT':'Tyler Toueg',\
        'JZ':'Jacob Ziontz'\
        }

Inspectors = []
total_images=0
counts_df = pd.DataFrame(columns=['Reviewer Initials','Review Date','Project','Tracer'])
for project in ACTIVE_PROJECTS[:]:
    print('Running for',project[0],project[1])
    # Get all the QC CSVs for a given project tracer
    files_ = glob.glob(f'/home/jagust/petcore/qc/{project[0]}/{project[1]}/QC_{project[0].upper()}_{project[1].upper()}_??-??-????.tsv')
    #concat_file = f'/home/jagust/petcore/qc/{project[0]}/{project[1]}/CONCAT_{project[0].upper()}_{project[1].upper()}.tsv'
    if len(files_) == 0:
        continue
    files_.sort(key=lambda date: datetime.strptime(re.search(r'_(\d{2}-\d{2}-\d{4}).tsv', date).group(1), "%m-%d-%Y"),reverse=True)
    #print(files_)
    # Load all QC df's and put them into a list. Merge the list of dfs, then drop duplicates by reviewer info.
    df_list=[]
    for df_i in files_:
        df_ = pd.read_csv(df_i,sep='\t',header=None,dtype=str,low_memory=False)
        df_.sort_values(by=0,inplace=True,ascending=False)
        HEAD = df_.iloc[0].values
        df_ = df_.iloc[1:]
        df_.columns = HEAD
        
        df_ = df_.set_index(['ID','TP','PET Date','MRI Date'])


        #df_ = df_.dropna(subset=['Reviewer Initials','Image Notes','QC Notes'],axis=0,how='all')
        df_list.append(df_)
    df_n = pd.concat(df_list,ignore_index=False,sort=True)
    df_n = df_n[~df_n.index.duplicated(keep='first')]
    total_images += len(df_n)

    filesn_ = glob.glob(f'/home/jagust/petcore/qc/{project[0]}/{project[1]}/NULL_{project[0].upper()}_{project[1].upper()}*.tsv')
    print('Searching null files:',f'/home/jagust/petcore/qc/{project[0]}/{project[1]}/NULL_{project[0].upper()}_{project[1].upper()}.tsv')
    print('Null Files:',filesn_)
    if os.path.exists(f'/home/jagust/petcore/qc/{project[0]}/{project[1]}/NULL_{project[0].upper()}_{project[1].upper()}.tsv'):
        filesn_df = pd.read_csv(filesn_[0],sep='\t',header=None,dtype=str,low_memory=False)
        filesn_df.sort_values(by=0,inplace=True,ascending=False)
        HEAD = filesn_df.iloc[0].values
        filesn_df = filesn_df.iloc[1:]
        filesn_df.columns = HEAD
        
        filesn_df = filesn_df.set_index(['ID','TP','PET Date','MRI Date'])
        df_n = df_n.append(filesn_df,ignore_index=True)
    #if os.path.exists(concat_file):
    #    df_concat = pd.read_csv(concat_file,sep='\t',dtype=str)
    #    df_n = df_n.append(df_concat,ignore_index=True)
    elif len(filesn_) > 1:
        #filesn_.sort(key=lambda date: datetime.strptime(re.search(r'_(\d{2}-\d{2}-\d{4}).tsv', date).group(1), "%m-%d-%Y"),reverse=True)
        csv_list = [pd.read_csv(i,sep='\t',header=None,dtype=str,low_memory=False) for i in filesn_]
        csv_list = [i.sort_values(by=0,ascending=False) for i in csv_list]
        tmp_ = []
        for i in csv_list:
            i.columns = i.iloc[0].values
            tmp_.append(i)
        csv_list = tmp_
        csv_list = [i.iloc[1:] for i in csv_list]
        csv_list = [i.set_index(['ID','TP','PET Date','MRI Date']) for i in csv_list]
        filesn_df = pd.concat(csv_list, axis=0, ignore_index=True)
        #filesn_df = filesn_df.drop_duplicates(ignore_index=True)
        filesn_df = filesn_df[~((filesn_df.index.duplicated(keep='first')) & (filesn_df.duplicated(keep='first')))]
        df_n = df_n.append(filesn_df,ignore_index=True)
    else:
        # There are no NULL files
        print('No NULL QC files for this project/tracer')
        #filesn_df = pd.read_csv(filesn_[0],sep='\t',dtype=str)

    df_n = df_n.replace(r'^\s*$', np.nan, regex=True)
    df_n = df_n.replace('',np.nan)
    df_n = df_n[~((df_n.index.duplicated(keep='first')) & (df_n.duplicated(keep='first')))]

    add_this = df_n[['Reviewer Initials','Review Date']].dropna(how='all')
    add_this = add_this.dropna(subset=['Review Date',],how='any')
    add_this.loc[add_this['Reviewer Initials'].isnull(),'Reviewer Initials'] = 'Unnamed'
    #add_this['Review Date'] = pd.to_datetime(add_this['Review Date'],infer_datetime_format=True)
    add_this['Reviewer Initials'] = add_this['Reviewer Initials'].str.upper()
    add_this['Reviewer Initials'] = add_this['Reviewer Initials'].str.strip()
    add_this['Project'] = project[0].upper()
    add_this['Tracer'] = project[1].upper()

    #counts_df=counts_df.append(add_this, ignore_index=True)

    counts_df = pd.concat([counts_df,add_this],ignore_index=True)

counts_df['Full Name'] = [names[i] if i in list(names.keys()) else i for i in counts_df['Reviewer Initials']]


color = '#303030'

def draw_barchart_plain(DATE_):
    
    dff = counts_df#(counts_df[counts_df['Review Date'] <= (DATE_)])#.sort_values(by='value', ascending=True)
    ax.clear()
   # ax.imshow(ocean_img,extent=[-5, 500, -5, len(names)])
    A,B=list(dff['Full Name'].value_counts().sort_values().reindex(counts_df['Full Name'].unique(), fill_value=0).sort_values().index),\
        list(dff['Full Name'].value_counts().sort_values().reindex(counts_df['Full Name'].unique(), fill_value=0).sort_values().values)
    #B[0]=670
    for person in list(names.values()):
        if person not in A:
            A.append(person)
            B.append(0)
    B,A = zip(*sorted(zip(B,A)))
    print(A)
    ax.barh(A,B,color=color,alpha=1)
    ax.set_facecolor('white')
    if np.nanmax(B) < 600:
        ax.set_xlim(-40,640)
        MAX = 600+40
    else:
        ax.set_xlim(-40,np.nanmax(B)+40)
        MAX = np.nanmax(B)+40
        
    # Background
   # ax.imshow(ocean_img,extent=[-45, MAX+1, -1, len(names)])
  #  ax.imshow(finish_img,extent=[496,460,-1,3])
    ax.axvline(x=500, color=color, linestyle='--',ymax=0.99,ymin=0.01,linewidth=3)
    ax.axvline(x=-5, color=color, linestyle='-',ymax=0.99,ymin=0.01,linewidth=3)
   # ax.text(460, -.75, "500", size=15, weight=600, ha='left',color=color)
    
    [x.set_linewidth(1.75) for x in ax.spines.values()]
   # dx = B.max() / 200
    for i, (value, name) in enumerate(zip(B,A)):
        #ax.text(value, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value, i,     f'{value:,.0f}',  size=14, ha='left',  va='center',color=color)
        height = 2
        #plt.imshow(squid_img, extent=[value-32, value - 2, i - height / 2, i + height / 2], aspect='auto', zorder=2)
    # ... polished styles
   # ax.text(1, 0.2, DATE_.strftime("%m/%Y"), transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    #ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
   # ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    #ax.set_yticks([])
    #ax.convert_units
    ax.margins(0, 0.01)
    #ax.grid(which='major', axis='x', linestyle='-')
  #  ax.set_axisbelow(True)
    qcd = np.nansum(B)
    ax.text(0, 1.01, f"Images QC'd on Squid    ({qcd} / {total_images})",
            transform=ax.transAxes, size=24, weight=600, ha='left')
    plt.box(False)
    #ax.spines.right.set_visible(False)
    #ax.spines.top.set_visible(False)
    #ax.spines.bottom.set_visible(False)
   # ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis='x', colors=color)
   # ax.set_xticks([])
    plt.tick_params(left = False)
    plt.tick_params(bottom = False)
    fig.patch.set_facecolor('#e3e3e3')


def draw_barchart(DATE_):
    
    dff = counts_df#(counts_df[counts_df['Review Date'] <= (DATE_)])#.sort_values(by='value', ascending=True)
    ax.clear()
   # ax.imshow(ocean_img,extent=[-5, 500, -5, len(names)])
    A,B=list(dff['Full Name'].value_counts().sort_values().reindex(counts_df['Full Name'].unique(), fill_value=0).sort_values().index),\
        list(dff['Full Name'].value_counts().sort_values().reindex(counts_df['Full Name'].unique(), fill_value=0).sort_values().values)
    #B[0]=670
    for person in list(names.values()):
        if person not in A:
            A.append(person)
            B.append(0)
    B,A = zip(*sorted(zip(B,A)))
    print(A)
    ax.barh(A,B,color=color,alpha=0)
    if np.nanmax(B) < 600:
        ax.set_xlim(-40,640)
        MAX = 600+40
    else:
        ax.set_xlim(-40,np.nanmax(B)+40)
        MAX = np.nanmax(B)+40
        
    # Background
    ax.imshow(ocean_img,extent=[-45, MAX+1, -1, len(names)])
  #  ax.imshow(finish_img,extent=[496,460,-1,3])
    ax.axvline(x=500, color='white', linestyle='--',ymax=0.99,ymin=0.01,linewidth=3)
    ax.text(460, -.75, "500", size=15, weight=600, ha='left',color='white')
    
    [x.set_linewidth(1.75) for x in ax.spines.values()]
   # dx = B.max() / 200
    for i, (value, name) in enumerate(zip(B,A)):
        #ax.text(value, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value, i,     f'{value:,.0f}',  size=14, ha='left',  va='center',color='white')
        height = 2
        plt.imshow(squid_img, extent=[value-40, value - 2, i - height / 2, i + height / 2], aspect='auto', zorder=2)
    # ... polished styles
   # ax.text(1, 0.2, DATE_.strftime("%m/%Y"), transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    #ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
   # ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    #ax.set_yticks([])
    #ax.convert_units
    ax.margins(0, 0.01)
    #ax.grid(which='major', axis='x', linestyle='-')
  #  ax.set_axisbelow(True)
    qcd = np.nansum(B)
    ax.text(0, 1.01, f"Images QC'd on Squid    ({qcd} / {total_images})",
            transform=ax.transAxes, size=24, weight=600, ha='left')
    plt.box(False)
   # ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticks([])
    plt.tick_params(left = False)
    plt.tick_params(bottom = False)
    fig.patch.set_facecolor('#e3e3e3')


    #ax.imshow(ocean_img,extent=[-5, 500, -5, len(names)])
    #finish_img
    
    
    
#tjw animate_dates = []
#tjw cur=counts_df['Review Date'].min()
#tjw plus=counts_df['Review Date'].min() +  timedelta(days=7)
#tjw while plus <= counts_df['Review Date'].max():
#tjw     plus+=timedelta(days=7)
#tjw     animate_dates.append(plus)
#tjw
#tjw
# Uncomment when enough data collected to make an animation
#tjw #print('Creating animation')
#tjw DPI=160
#tjw W=15
#tjw H=8
#tjw fig, ax = plt.subplots(figsize=(W,H),dpi=DPI)
#tjw animator = animation.FuncAnimation(fig, draw_barchart, frames=animate_dates)
#tjw #HTML(animator.to_jshtml()) 
#tjw print('Saving animation','../images/Who_is_QCing.gif')
#tjw animator.save('../images/Who_is_QCing.gif', writer='imagemagick', fps=10)

print(counts_df)
outp = '../images/Who_is_QCing.png'
if os.path.exists(outp):
    subprocess.run(['mv',outp,'../images/Who_is_QCing_last.png'])

print('Saving png','../images/Who_is_QCing.png')
fig, ax = plt.subplots(figsize=(11, 8),dpi=140,linewidth=6,edgecolor='#141414')
draw_barchart(counts_df['Review Date'].max())
fig.savefig('../images/Who_is_QCing.png',edgecolor=fig.get_edgecolor(),bbox_inches='tight')
fig, ax = plt.subplots(figsize=(11, 8),dpi=140,linewidth=6,edgecolor='#141414')
draw_barchart_plain(counts_df['Review Date'].max())
fig.savefig('../images/Who_is_QCing_plain.png',edgecolor=fig.get_edgecolor(),bbox_inches='tight')


print('Done')
