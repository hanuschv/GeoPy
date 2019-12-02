# ==================================================================================================== #
#   Assignment_07                                                                                      #
#   (c) Vincent Hanuschik, 26/11/2019                                                                  #
#                                                                                                      #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import ogr
import pandas as pd
import numpy as np

# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/'
country_data = dir + 'Assignmet_07_data/SouthAmerica/gadm_SouthAmerica.shp'
pa_data = dir +  'Assignmet_07_data/WDPA_May2019_shapefile_polygons.shp'
s_america = ogr.Open(country_data)
pa_glob = ogr.Open(pa_data)

lyrc = s_america.GetLayer()
lyrc_names = [field.name for field in lyrc.schema]

lyr_pa = pa_glob.GetLayer()
lyrpa_names = [field.name for field in lyr_pa.schema]

# 'terrestrial (Marine = 0), Status: designated, Category: IUCN_CAT all; STATUS_YR; REP_AREA; NAME'

#create data frame for results
summary = pd.DataFrame(columns={'Country ID':[],
                                'Country Name':[],
                                'PA Category':[],
                                '# PAs':[],
                                'Mean area of PAs':[],
                                'Area of largest PA':[],
                                'Name of largest PA':[],
                                'Year of establ. Of largest PA':[]
                                })
#load layer of protected areas
lyr_pa = pa_glob.GetLayer()
lyr_pa.SetAttributeFilter("STATUS IN ('Designated') AND MARINE IN ('0') "
                         "AND IUCN_CAT IN ('Ia', 'Ib', 'II','III', 'IV', 'V', 'VI')")


# TO IMPROVE:
#       - the resulting dataframe has the wrong order within the category column

#country id iterator, increase at the end of loop
i = 1
for country in lyrc:            # loop through all countries
    #get the name of the country
    country_name = country.GetField('NAME_0')
    print(country_name)
    # set spatial filter for country
    lyr_pa.SetSpatialFilter(country.geometry())
    # get the number of PA for the country
    pa_count = lyr_pa.GetFeatureCount()
    # creaty empty arrays with the shape equal to the number of features in that country
    PA_DEF = np.empty(pa_count, dtype='U25')
    area = np.empty(pa_count)
    names = np.empty(pa_count, dtype='U25')
    years = np.empty(pa_count, dtype=int)
    #feature index, , increase at the end of loop
    j = 0
    #loop through all features per country and write all attributes into the arrays
    for pa in lyr_pa :
        PA_DEF[j] = pa.GetField('IUCN_CAT')
        area[j] = pa.GetField('REP_AREA')
        names[j] = pa.GetField('NAME')
        years[j] = pa.GetField('STATUS_YR')
        j += 1
    #reset spatial filter
    lyr_pa.SetSpatialFilter(None)

    # add entry in the data frame with respective statistics of pa's per country
    summary = summary.append({'Country ID': i,
                            'Country Name': country_name,
                            'PA Category': "all",
                            '# PAs': pa_count,
                            'Mean area of PAs': np.mean(area),
                            'Area of largest PA': max(area),
                            'Name of largest PA': names[np.argmax(area)],
                            'Year of establ. Of largest PA': years[np.argmax(area)]}, ignore_index=True)

    # loop through pa categories
    for cat in np.unique(PA_DEF):
        #filter category
        cat_area = area[PA_DEF == cat]
        # add entry in the data frame with statistics for each pa category
        summary = summary.append({'Country ID': i,
                        'Country Name': country_name,
                        'PA Category': cat,
                        '# PAs': len(cat_area),
                        'Mean area of PAs': np.mean(cat_area),
                        'Area of largest PA': max(cat_area),
                        'Name of largest PA': names[PA_DEF == cat][np.argmax(cat_area)],
                        'Year of establ. Of largest PA':  years[PA_DEF == cat][np.argmax(cat_area)]}, ignore_index=True)
    i += 1

# lyr_pa.ResetReading()
# lyrc.ResetReading()

summary.to_csv("pa_south_america_summary.csv")

# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")