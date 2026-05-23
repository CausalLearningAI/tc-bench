import requests
from bs4 import BeautifulSoup
import wget
import os
import numpy as np
import pandas as pd
import sys 

#code to download gridsat brightness temperature matrices (netcdf format) from the website
#progress and errors of the process are saved in  'log_download.txt' and 'log_ERRORS.txt' respectively 


year=sys.argv[1]  #download is parallelized across the years
directory_ibtracs=''
ibtracs_filename='ibTRACS_since_1980.csv' #csv containing the cyclone tracks
output_directory=''
os.makedirs
df=pd.read_csv(os.path.join(directory_ibtracs+ibtracs_filename))

times=df['ISO_TIME'].loc[df['ISO_TIME'].str.contains(year)]
times=times.str.replace('-','.')
times=times.str.split(':').str[0]
times=times.str.replace(' ','.')
times=np.unique(times)

url = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/"+year+'/'

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
file_extensions = (".nc")
links = []

for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(file_extensions):
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(url+href)
            

    #print(links)
for link in links:
        file=os.path.join(output_directory,link.split('/')[-1])
        #print(file)
        with open(os.path.join(output_directory,'log_download.txt'), 'a') as f:
                    f.write(file)
                    f.write('\n')
                    f.close()
        if any(time in link for time in times) and os.path.exists(file)==False:
            
            try:
                with open(os.path.join(output_directory,'log_download.txt'), 'a') as f:
                    f.write('downloading   '+link)
                    f.write('\n\n')
                    f.close()
   
                wget.download(link, out=output_directory)
            
            except:
                with open(os.path.join(output_directory,'log_ERRORS.txt'), 'a') as f:
                    f.write('failed downloading   '+link)
                    f.write('\n\n')
                    f.close()
                continue    
        else:
            with open(os.path.join(output_directory,'log_download.txt'), 'a') as f:
                    f.write('file already there '+file)
                    f.write('\n')
                    f.close()
            continue
