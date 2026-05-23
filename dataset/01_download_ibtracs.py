import wget
import os 
directory_ibtracs=''

url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv'


file_name = 'ibTRACS_since_1980.csv'
wget.download(url, os.path.join(directory_ibtracs+file_name))
