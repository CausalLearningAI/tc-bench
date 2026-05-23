import numpy as np
import pandas as pd
import os 

path_ibtracs=''


df=pd.read_csv(os.path.join(path_ibtracs,'ibTRACS_since_1980.csv'))
print('number of available cyclones',len(np.unique(df['SID'].values)))


#some processing to clean the first row of the dataset.
#it mostly contains measure units of some variables, that are included in the header,like 'WIND kts' or 'PRES mb'

first_row=[str(df.iloc[0].values[i]).strip() for i in range (len(df.iloc[0].values))]
df.columns = [
    (df.columns[i] + ' ' + first_row[i]).strip() if 'degrees' not in first_row[i]
    else df.columns[i].strip()
    for i in range(len(df.columns))
]

#drop the first row after this processing
df=df.iloc[1:].reset_index(drop=True) 

#ensure that wind, pressure and lat-lon columns are numeric
wind_cols=[]
lat_lon=[]
lat_lon=list(df.columns[np.logical_or(df.columns.str.contains('LAT'),df.columns.str.contains('LON'))])
wind_press=list(df.columns[np.logical_or(df.columns.str.contains('WIND'),df.columns.str.contains('PRES'))])
        
for col in wind_press+lat_lon:
    df[col]=pd.to_numeric(df[col],errors='coerce')


agencies=list(np.unique(df['WMO_AGENCY'].values))
agencies = [a.strip() for a in agencies if a.strip()]



agency_capitals=['USA','BOM','USA','USA','USA','NADI','NEWDELHI','REUNION','TOKYO','WELLINGTON']
print(agencies,agency_capitals)


#Only columns that are interesting/useful for the following analysis are kept:
#Those include some general information on the track, the time and location info, and the wind and pressure data

cols_to_keep=['SID','SEASON Year','NUMBER','BASIN','SUBBASIN','NAME','ISO_TIME','NATURE','LAT','LON']

for i in range (len(agencies)):
    print(agencies[i])
    if agencies[i]=='cphc': #WMO provides very little pressure data for this agency, so it's disregarded
        continue
    df_save=df.loc[df['WMO_AGENCY']==agencies[i]].reset_index(drop=True)

    col_keep_agency=cols_to_keep+list(df.columns[df.columns.str.contains(agency_capitals[i])])
    col_keep_agency=col_keep_agency+list(df.columns[df.columns.str.contains('WMO')])
    df_save=df_save[col_keep_agency]
    print(col_keep_agency)
    print('len dataset',len(df_save))
    #if i!=0:
        #leng=leng+len(np.unique(df_save['SID'].values))
    print('number cyclones',len(np.unique(df_save['SID'].values)),'\n\n')
    df_save.to_csv(os.path.join(path_ibtracs,'dataset_ibtracs_basic_cols_'+agencies[i]+'.csv'))
    print(df_save['WMO_PRES mb'].mean())




# df_nonempty = df[df["WMO_AGENCY"].str.strip() != ""] #a few cyclones are not attributed to any agency, so they are disregarded

# df_nonempty_grouped = (df_nonempty.groupby("SID")["WMO_AGENCY"].agg(lambda x: list(x.unique())).reset_index()) #group the rows belonging to the same cyclone (SID is the identifier) and transform the agency entry in a list
# #some of the rows will have more than one agency, so in this case the length of this list will be >1

# #only the ones with one single agency are kept
# df_one_agency_grouped = df_nonempty_grouped.loc[df_nonempty_grouped["WMO_AGENCY"].str.len()==1] 

# #back to the ungrouped dataset, keeping the entries of the cyclones belonging to the one-agency dataset

# df_one_agency=df.loc[df['SID'].isin(df_one_agency_grouped['SID'].values)].reset_index(drop=True)
