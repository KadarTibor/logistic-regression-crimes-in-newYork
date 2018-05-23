import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model

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
# print(dataset.head())

# pre process the data
features = dataset[['CMPLNT_FR_DT',
                    'CMPLNT_FR_TM',
                    'OFNS_DESC',
                    'BORO_NM']]

# convert the date column into month column
date = dataset.CMPLNT_FR_DT
date = date.fillna('12/04/2015')
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

# encode the data (we have strings as labels want numerical values)

one_hot_date = pd.get_dummies(features.CMPLNT_FR_DT)
one_hot_time_of_day = pd.get_dummies(features.CMPLNT_FR_TM)
one_hot_offense_description = pd.get_dummies(features.OFNS_DESC)
one_hot_borough = pd.get_dummies(features.BORO_NM)

features.drop('CMPLNT_FR_DT', axis=1, inplace=True)
features.drop('CMPLNT_FR_TM', axis=1, inplace=True)
features.drop('OFNS_DESC', axis=1, inplace=True)
features.drop('BORO_NM', axis=1, inplace=True)
features = features.join(one_hot_date)
features = features.join(one_hot_time_of_day)
features = features.join(one_hot_offense_description)
features = features.join(one_hot_borough)

# transform the labels into a dataframe to join with the dataset pre split
labels = pd.DataFrame(labels.reshape(len(labels), 1), ["LABL"])

features = features.join(labels)

# split the dataset into train validation and test sets
# randomly shuffle the dataframe
features = features.reindex(np.random.permutation(features.index))
# split the data set into train validate and test sets 60% 20% 20%
train, validate, test = np.split(features.sample(frac=1), [int(.6*len(features)), int(.8*len(features))])

print(train)

# apply the learning algorithm on the data set
# clf = linear_model.LogisticRegression(C=1e5)
# clf.fit(features, labels)


