from __future__ import division #for float division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

#Disable chained assignments error
pd.options.mode.chained_assignment = None

#reading data from csv files
user_data = pd.read_csv('../data/datamining_user_data.csv')
trans_data = pd.read_csv('../data/datamining_transaction_data.csv')
contract = pd.read_csv('../data_new/datamining_contract_data.csv')

def status_fill(row):
    if row['non_performing'] == True:
        a = 'Default'
    elif row['fully_paid_contract'] == True:
        a = 'Fully Paid'
    else:
        a = 'Performing'
    return a

contract['status'] = contract.apply(lambda row : status_fill(row), axis=1)
contract['duration'] = pd.to_numeric(contract['duration'])

#Calculate detault rates per contract and per borrower
FP = len(contract[contract['status'] == 'Fully Paid'])
NP = len(contract[contract['status'] == 'Default'])
P = len(contract[contract['status'] == 'Performing'])
print 'default rate per borrower: %f percent' %round(70/233*100,2)
print 'default rate per loan: %f percent' %round(71/308*100,2)

#Now let's take a look at some breakdowns

#Loan by status
fig = plt.figure(figsize=(7, 5))
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
xdata = ['Performing', 'Fully Paid', 'Default']
ydata = [121, 44, 68]
sns.barplot(x=xdata, y = ydata)
plt.title('Small Business Loan Status', fontsize=18)
plt.xlabel('')#, fontsize = 14)
plt.ylabel('no of small businesses', fontsize = 14)
plt.show()

#Small businesses with repeated loans
fig = plt.figure(figsize=(7, 5))
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
sns.countplot(x=group_2['funded_at'])
plt.title('Small Businesses with Repeated Loans', fontsize=18)
plt.xlabel('no of loans per small business', fontsize = 14)
plt.ylabel('no of small businesses', fontsize = 14)
plt.show()

#Loan duration
fig = plt.figure(figsize=(7, 5))
plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14})
plt.hist(contract['duration'], bins = 5) #, bins = 6, color = 'b')
plt.title('Loan Duration', fontsize=18)
plt.xlabel('days', fontsize = 14)
plt.ylabel('no of loans', fontsize = 14)
plt.show()

#Loan amount
fig = plt.figure(figsize=(7, 5))
plt.hist(contract['funded_amount']) #, bins = 6, color = 'b')
plt.title('Loan Amount', fontsize=18)
plt.xlabel('$ amount', fontsize = 14)
plt.ylabel('no of loans', fontsize = 14)
plt.xticks
plt.show()

#Filtering users who were funded
user_funded = user[user['funded_amount'].isnull() == False]
user_funded['duration'] = user_funded['duration'].str[:3]
user_funded['duration'] = pd.to_numeric(user_funded['duration'])
user_funded['status'] = user_funded.apply(lambda row : status_fill(row), axis=1)

#Add 2-digit NAICS code column
user_funded['naics_code'] = user_funded['naics_code'].astype(str)
user_funded['2-digit'] = user_funded['naics_code'].str[:2].astype(int)
code_map = pd.read_csv('../data_new/naics_code_mapping.csv')
user_funded = user_funded.merge(code_map)
user_funded.to_csv('../data_new/user_funded.csv')

#Manually checking business activity description field, to ensure that industry code is correct
user_correct = pd.read_csv('../data_new/user_corrected.csv')
del user_correct['industry_group']
user_correct = user_correct.merge(code_map)
user_correct.set_index('user_id')
user_correct['industry_group'].replace('Administrative and Support and Waste Management and Remediation Services', 'Administrative, Support and Waste Management', inplace=True)

#Let's look at default rates per industry
industry_list = user_correct['industry_group'].unique()
FP_list = np.empty(len(industry_list))
NP_list = np.empty(len(industry_list))
P_list = np.empty(len(industry_list))
for no in range(len(new_ind_list)):
    f = len(user_correct[(user_correct['industry_group'] == industry_list[no]) & (user_correct['status'] == 'Fully Paid')])
    FP_list[no] = f
    n = len(user_correct[(user_correct['industry_group'] == industry_list[no]) & (user_correct['status'] == 'Default')])
    NP_list[no] = n
    p = len(user_correct[(user_correct['industry_group'] == industry_list[no]) & (user_correct['status'] == 'Performing')])
    P_list[no] = p

NP_pt = NP_list/(FP_list + NP_list + P_list)*100
dict = {}
new_ind_list = []
for no in range(len(industry_list)):
    dict[industry_list[no]] = NP_pt[no]
temp_list = sorted(dict.items(), key=lambda x: x[1], reverse = False)
for ind, pt in temp_list:
    new_ind_list.append(ind)

FP1_list = np.empty(len(new_ind_list))
NP1_list = np.empty(len(new_ind_list))
P1_list = np.empty(len(new_ind_list))
for no in range(len(new_ind_list)):
    f = len(user_correct[(user_correct['industry_group'] == new_ind_list[no]) & (user_correct['status'] == 'Fully Paid')])
    FP1_list[no] = f
    n = len(user_correct[(user_correct['industry_group'] == new_ind_list[no]) & (user_correct['status'] == 'Default')])
    NP1_list[no] = n
    p = len(user_correct[(user_correct['industry_group'] == new_ind_list[no]) & (user_correct['status'] == 'Performing')])
    P1_list[no] = p

#Show horisontal stacked bar chart with default status per industry, sorted from highest to lowest
fig = plt.figure(figsize=(14, 7))
ind = np.arange(len(industry_list))
plt.suptitle('Loan Status per Industry Group', fontsize=24)
plt.title('(ordered by % of defaults, highest to lowest)', fontsize = 18)
p1 = plt.barh(ind, FP1_list, color='g')
lefts = FP1_list
p2 = plt.barh(ind, P1_list, color = 'b', left=lefts)
lefts = FP1_list+P1_list
p3 = plt.barh(ind, NP1_list, color = 'r', left=lefts)
plt.legend(labels = ['Fully Paid', 'Performing', 'Default'], fontsize = 18, bbox_to_anchor=(1.1, 1.05), fancybox = True)
plt.yticks(ind, new_ind_list, rotation = 360)
plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16})
plt.ylabel('', fontsize = 16)
plt.xlabel('no of small businesses', fontsize = 18)
plt.show()

#Creating user_copy DF with status = first status
user_copy = user_correct[[u'user_id', u'user_created_at', u'first_funded_at',
       u'last_funded_at', u'funded_amount', u'duration',
       u'fully_paid_contract', u'non_performing', u'zip_code', u'naics_code',
       u'industry', u'2-digit', u'business_activities', u'status',
       u'industry_group']]
user_copy['first_status'] = user_copy['status']
double = group_2[group_2['funded_at']>1]
double_list = double.index.unique()
group = contract.groupby('user_id').min()
user_copy = user_copy.set_index('user_id')

for user in double_list:
    fdate = user_copy.loc[user]['first_funded_at']
    #print 'user is', user
    #print 'fdate is', fdate
    row = contract[(contract['user_id'] == user) & (contract['funded_at'] == fdate)]
    #print 'status is', row['status'].item()
    user_copy.at[user, 'first_status'] = row['status'].item()
user_copy['first_status'].replace('Non Performing', 'Default', inplace=True)

#Creating time series
funded_users = user_copy[user_copy.index.isin(trans_data['user_id'].unique())]
user_list = funded_users.index
trans_f = trans_data[trans_data['user_id'].isin(user_list)]
trans_f = trans_f.sort_values(['user_id', 'transacted_on'])
funded_users['first_funded_at'] = pd.to_datetime(funded_users['first_funded_at'])
trans_f['transacted_on'] = pd.to_datetime(trans_f['transacted_on'])
columns_dt = ['first_funded_at', 'first_transaction', 'last_transaction',
                'ts_last_date', 'ts_length']
dt_compare = pd.DataFrame(index = user_list, columns = columns_dt)

for user in user_list:
    fund_date = funded_users.loc[user]['first_funded_at']
    first_trans_date = trans_f[trans_f['user_id'] == user]['transacted_on'].min()
    last_trans_date = trans_f[trans_f['user_id'] == user]['transacted_on'].max()

    #Calculate how many days of transaction history for each user
    day_diff = pd.to_timedelta(-1, unit = 'd')
    ts_last_date = min((fund_date + day_diff), last_trans_date)
    ts_length = max((ts_last_date - first_trans_date).days, 0)

    #Fill in the dates for each user
    dt_compare.loc[user]['first_funded_at'] = fund_date
    dt_compare.loc[user]['first_transaction'] = first_trans_date
    dt_compare.loc[user]['last_transaction'] = last_trans_date
    dt_compare.loc[user]['ts_last_date'] = ts_last_date
    dt_compare.loc[user]['ts_length'] = ts_length

#Calculate no of users with ts_length >= 30, 60 and 90 days
users_30 = dt_compare[dt_compare['ts_length'] >= 30]
users_60 = dt_compare[dt_compare['ts_length'] >= 60]
users_90 = dt_compare[dt_compare['ts_length'] >= 90]

print 'Total no of users: ', len(user_list)
print 'Users length >= 30 days: %d, which is %d percent of total' % (len(users_30), round(len(users_30)/len(user_list)*100,2))
print 'Users length >= 60 days: %d, which is %d percent of total' % (len(users_60), round(len(users_60)/len(user_list)*100,2))
print 'Users length >= 90 days: %d, which is %d percent of total' % (len(users_90), round(len(users_90)/len(user_list)*100,2))

#Plot the histogram of prior_days with vertical lines showing 30d, 60d and 90d
plt.hist(dt_compare['ts_length'], bins = dt_compare['ts_length'].max(), color = 'b')
plt.title('Transaction history length prior to funding per user, in days')
plt.xlabel('days')
plt.ylabel('users')
plt.axvline(x=30, color = 'r', linestyle = '--')
plt.axvline(x=60, color = 'r', linestyle = '--')
plt.axvline(x=90, color = 'r', linestyle = '--')
plt.show()

#Create DF of time series for all users who have >= N days transcaction data
N = 31
users_N = dt_compare[dt_compare['ts_length'] >= N]
days = [np.arange(-N, 1)]
#days = np.array([np.arange(-N, 1)]).T
#ts_N = pd.DataFrame(index = days, columns = sers_N.index)
ts_N = pd.DataFrame(index = users_N.index, columns = days) #initiating empty DF

#Now fill DF with actual time series
for user in ts_N.index:
    cumsum = 0
    for delta in ts_N.columns:
        day = (users_N.loc[user]['ts_last_date'] + pd.to_timedelta(delta, unit = 'd'))#[0]
        day_subset = trans_f[(trans_f['user_id'] == user) & (trans_f['transacted_on'] == day)]
        day_amount = day_subset['amount'].sum()
        cumsum += day_amount
        ts_N.at[user, delta] = cumsum

#Let's merge the time series DF with information from funded_users DF
ts_N = ts_N.join(funded_users, how = 'left')
col_to_drop = ['user_created_at','fully_paid_contract', 'non_performing', 'naics_code', 'industry', 'business_activities']
ts_N = ts_N.drop(col_to_drop, axis = 1)
ts_N['target'] = np.where(ts_N['first_status'] == 'Default', 1, 0)

#Normalize time series by dividing the changes in daily bank balance by requested loan amount
for col in range(-31, 1):
    ts_N[col] = ts_N[col]/ts_N['funded_amount']
ts_N.to_csv('../data_new/ts_N.csv')

#Load data with derived features and fit Logistic regression model
raw_data = pd.read_csv('../data_new/raw_data_regression.csv')
y_data = raw_data['target']

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
import statsmodels.api as sm

dummies_state = pd.get_dummies(raw_data['state'], prefix = 'state', drop_first=True)
dummies_industry = pd.get_dummies(raw_data['industry_group'], prefix = 'ind', drop_first=True)
raw_data = pd.concat([raw_data, dummies_state, dummies_industry], axis=1)

raw_data.drop(raw_data.columns[['user_id', 'industry_group', 'state', 'target', 'first_funded_at']], axis=1, inplace=True)
x_data = raw_data
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=1, stratify=y_data)

model = LogisticRegression()
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model.score(X_test, y_test)

TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_predict).ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print 'Accuracy, presision, recall:', accuracy, precision, recall

scores = cross_val_score(model, x_data, y_data, scoring='accuracy', cv=50)
print scores
print scores.mean()

cols = X_train.columns.tolist()
coef = model.coef_.tolist()[0]
dict = {}
for col, imp in zip(cols, coef):
    dict[col] = imp
temp_list = sorted(dict.items(), key=lambda x: x[1], reverse = True)

#Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(class_weight = 'balanced_subsample')
model2.fit(X_train, y_train)
model2.score(X_test, y_test)
y_predict2 = model2.predict(X_test)
feat_imp = model2.feature_importances_.tolist()
TN2, FP2, FN2, TP2 = metrics.confusion_matrix(y_test, y_predict2).ravel()
accuracy2 = (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2)
precision2 = TP2/(TP2+FP2)
recall2 = TP2/(TP2+FN2)
print 'Accuracy, presision, recall:', accuracy2, precision2, recall2

dict2 = {}
for col, imp in zip(cols, feat_imp):
    dict2[col] = imp
temp2_list = sorted(dict2.items(), key=lambda x: x[1], reverse = True)
temp2_list[:15]

scores2 = cross_val_score(model2, x_data, y_data, scoring='accuracy',cv=50)
print scores2
print scores2.mean()
