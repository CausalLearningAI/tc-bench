import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import traceback
#year=sys.argv[1]


#


agencies=['atcf', 'bom', 'hurdat_atl', 'hurdat_epa', 'nadi',
       'newdelhi', 'reunion', 'tokyo', 'wellington']
agencies_capital=['USA','BOM','USA','USA','NADI','NEWDELHI','REUNION','TOKYO','WELLINGTON']

agency_index=int(sys.argv[1])  #parallelization across the different agencies
agency=agencies[agency_index]

path_output=''
path_gridsat_data=''
path_ibtracs=''



df_tot=pd.read_csv(os.path.join(path_ibtracs,'dataset_ibtracs_basic_cols_'+agency+'.csv'))
var_press=agencies_capital[agency_index]+'_PRES mb'

df_tot[var_press]=pd.to_numeric(df_tot[var_press],errors='coerce')
var_wind=agencies_capital[agency_index]+'_WIND kts'

df_tot[var_wind]=pd.to_numeric(df_tot[var_wind],errors='coerce')

#IBTRACS use both the -180-180 and the 0-360 longitude formats depending on the agency
#here all the longitude values are transformed into 

df_tot['LON']=(df_tot['LON'].values + 180) % 360 - 180

for year in [str(t) for t in range (1980,2025)]:
    
    df_year=df_tot.loc[df_tot['ISO_TIME'].str.contains(year)].reset_index(drop=True)
    if len(df_year)==0:
        continue
    names=np.unique(df_year['NAME'].values)
    half_width=7.5
    half_width_points=112
    for name in names:
                
            df=df_year.loc[df_year['NAME']==name].reset_index(drop=True)
            os.makedirs(os.path.join(path_output,year), exist_ok=True)
            print(os.path.exists(os.path.join(path_output,year)))
            print(os.path.join(path_output,year))
            #in case a previous download was interrupted
            lst = os.listdir(os.path.join(path_output,year))#+str(year)+'/')
            if len(lst)>0:
                lst.sort()
                lst=[lst[i] for i in range (len(lst)) if '.nc' in lst[i]]
                if len(lst)>0:    
                    os.remove(os.path.join(path_output,year,lst[-1]))   
                    
            times2=df['ISO_TIME']
            times=df['ISO_TIME'].str.split(':').str[0]
            times=times.str.replace(' ','.')
            times=times.str.replace('-','.')
            j=-1
            cyclone_full=[]
            for time1 in times:
                #print(time1)
            #print(time)
                j+=1
                file=path_gridsat_data+year+'/GRIDSAT-B1.'+time1+'.v02r01.nc'
                
                #print(file)
                try:
                    
                    with open(os.path.join(path_output,year,'log_download.txt'), 'a') as f:
                            f.write('opening '+file)
                            f.write('\n')
                            f.close()
                    ds=xr.open_dataset(file)
                    lat_ds=ds.lat.values
                    lon_ds=ds.lon.values
                    grid_spacing_lat=np.nanmean(np.diff(lat_ds))
                    grid_spacing_lon=np.nanmean(np.diff(lon_ds))
                    
                    lat_cen=df['LAT'][df['ISO_TIME']==times2.iloc[j]].values[0]
                    lon_cen=df['LON'][df['ISO_TIME']==times2.iloc[j]].values[0]
                    lat_cen_index=np.argmin(np.abs(lat_ds-lat_cen))
                    lon_cen_index=np.argmin(np.abs(lon_ds-lon_cen))
                    nx=len(lon_ds)


                    #the selection of the points in the longitudinal direction must be periodic across the boundary
                    lon_sel=(np.arange(lon_cen_index - half_width_points, lon_cen_index+half_width_points) % nx)
                    lat_sel=(np.arange(lat_cen_index - half_width_points, lat_cen_index+half_width_points))

                    pres=df['LON'][df['ISO_TIME']==times2.iloc[j]].values[0]
                    
                    cycl=ds['irwin_cdr'][0].isel(lon=xr.DataArray(lon_sel, dims="lon"),lat=xr.DataArray(lat_sel, dims="lat"))
                    
                    #lat and lon are redefined as difference from the coordinates of the center
                    new_lat=np.arange(-half_width_points*grid_spacing_lat,half_width_points*grid_spacing_lat,grid_spacing_lat)
                    new_lon=np.arange(-half_width_points*grid_spacing_lon,half_width_points*grid_spacing_lon,grid_spacing_lon)

                    cycl = cycl.assign_coords(lat=new_lat,lon=new_lon)
                    cycl.attrs[var_press] = df[var_press][df['ISO_TIME']==times2.iloc[j]].values[0]
                    cycl.attrs[var_wind] = df[var_wind][df['ISO_TIME']==times2.iloc[j]].values[0]
                    cycl.attrs['time']=time1
                    cycl = cycl.expand_dims(timestep=[j])
                    np_cycl=cycl.values
                    cycl.attrs['fraction nan']=np.count_nonzero(np.isnan(np_cycl)==True)/len(np_cycl.flatten())

                    #shape of (lat,lon) is supposed to be (224,224) for every snapshot
                    if np.shape(cycl)!=(1,224,224):
                        with open(os.path.join(path_output,'log_ERRORS.txt'),'a') as f:
                            f.write('different shape '+name+'   '+year+' '+time1)
                            f.write('  ')
                            f.write('shape '+str(np.shape(cycl)))
                            f.write('  ')
                            f.write('lat cen  '+str(lat_cen)+' lon cen '+str(lon_cen))
                            f.write('\n')
                            
                    elif cycl.attrs['fraction nan']>0.2:
                        with open(os.path.join(path_output,'log_ERRORS.txt'),'a') as f:
                            f.write('many nans '+name+'   '+year+' '+time1)
                            f.write('  ')
                            f.write(str(cycl.attrs['fraction nan']))
                            f.write('  ')
                            f.write('lat cen  '+str(lat_cen)+' lon cen '+str(lon_cen))
                            f.write('\n')        
                    
                    #if os.path.exists(path_output+year+'_'+name+'/'+name+'_'+time1+'.nc')==False:
                    cyclone_full.append(cycl)
                    #cycl.to_netcdf(path_output+year+'_'+name+'/'+name+'_'+time1+'.nc')
                        #print(path_output+year+'_'+name+'/'+name+'_'+time1+'.nc')
                        
                except Exception as e:
                    with open(os.path.join(path_output,'log_ERRORS.txt'),'a') as f:
                        f.write(year+'_'+name+time1)
                        f.write('\n')
                        if hasattr(e, "filename") and e.filename is not None:
                            f.write(f"{type(e).__name__}: {e.filename}\n")
                        else:
                            f.write(f"{type(e).__name__}: {e}\n")
                      
                    continue

            cyclone_full=xr.concat(cyclone_full[:],dim='timestep')
            cyclone_full.to_netcdf(os.path.join(path_output,year,name+'.nc'))
