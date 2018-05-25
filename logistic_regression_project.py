import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn import metrics


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


def compute_labels(offense_name):
    offense_name = str(offense_name)
    if offense_name == 'MURDER & NON-NEGL. MANSLAUGHTER':
        return 1
    if offense_name == 'HOMICIDE-NEGLIGENT,UNCLASSIFIED':
        return 1
    if offense_name == 'ASSAULT 3 & RELATED OFFENSES':
        return 1
    if offense_name == 'ROBBERY':
        return 1
    if offense_name == 'RAPE':
        return 1
    return 0


def calc_pozitive_negative_ratio(data):
    good = data[(data.Labels == 1)]
    nr_good= len(good)
    print('murders: %d' % nr_good)
    print('all_crimes: %d' % len(data.Labels))

    print('Percentage: %f' % ((nr_good / (len(data.Labels))) * 100))


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

features.CMPLNT_FR_DT = date
features.CMPLNT_FR_TM = time

labels = pd.DataFrame()
labels['Labels'] = dataset.OFNS_DESC.apply(lambda x: compute_labels(x))

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

# add the labels to the data frame
features = features.join(labels)

# since the data set contains more negative examples than positive
# we need to balance this out
positive_data = features[features.Labels == 1]
negative_data = features[features.Labels == 0]

# choose randomly negative data
negative_data = negative_data.sample(n=len(positive_data))

features = pd.concat([positive_data, negative_data])
# split the dataset into train validation and test sets
# randomly shuffle the dataframe
features = features.reindex(np.random.permutation(features.index))
# split the data set into train validate and test sets 60% 20% 20%
train, validate, test = np.split(features.sample(frac=1), [int(.6*len(features)), int(.8*len(features))])

# compute the pozitive and negative examples ratio
print('All the dataset')
calc_pozitive_negative_ratio(features)
print('Train dataset')
calc_pozitive_negative_ratio(train)
print('Validation dataset')
calc_pozitive_negative_ratio(validate)
print('Test dataset')
calc_pozitive_negative_ratio(test)
# apply the learning algorithm on the data set
clf = linear_model.LogisticRegression(C=0.0001)
clf.fit(train.loc[:, train.columns != 'Labels'], train.Labels)

score = clf.score(validate.loc[:, validate.columns != 'Labels'], validate.Labels)
print("score on validate set")
print(score)

score = clf.score(test.loc[:, test.columns != 'Labels'], test.Labels)
print("score on test set")
print(score)







