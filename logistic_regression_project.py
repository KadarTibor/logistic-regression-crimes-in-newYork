import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def get_month(date):
    tokens = str(date).split('/')
    return tokens[0]


def process_time(time):
    tokens = str(time).split(':')
    if int(tokens[0]) <= 6:
        return 'night'
    if int(tokens[0]) <= 12:
        return 'before_noon'
    if int(tokens[0]) <= 18:
        return 'after_noon'
    if int(tokens[0]) <= 24:
        return 'evening'


# Load the file containing the crimes in new york data
dataset = pd.read_csv("crimes-new-york-city/NYPD_Complaint_Data_Historic.csv", low_memory=False)

# print the header of the table representing the data
print(dataset.head())

# pre process the data
features = dataset[['CMPLNT_FR_DT',
                    'CMPLNT_FR_TM',
                    'OFNS_DESC',
                    'BORO_NM']]

# convert the date column into month column
date = dataset.CMPLNT_FR_DT
date = gdate.fillna('12/04/2015')
date = date.apply(lambda x: get_month(x))

# split the time column into day regions
time = dataset.CMPLNT_FR_TM
time = time.fillna('22:00:00')
time = time.apply(lambda x: process_time(x))
# print(time)

features.CMPLNT_FR_DT = date
features.CMPLNT_FR_TM = time

print(features)

labels = np.where((features.OFNS_DESC == 'MURDER & NON-NEGL. MANSLAUGHTER') |
                  (dataset.OFNS_DESC == 'HOMICIDE-NEGLIGENT,UNCLASSIFIE') |
                  (dataset.OFNS_DESC == 'ASSAULT 3 & RELATED OFFENSES'), 1, 0)

print(labels)