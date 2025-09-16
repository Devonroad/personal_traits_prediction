#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from model_wrapper import TestModel

# parameters
learning_rate=0.1
max_depth=3
min_samples_leaf=3
n_splits=10
random_state=42
output_file = 'model.bin'

# preprocessing data
print('Preprocessing data...')
df = pd.read_csv('./data.csv')

df.columns = df.columns.str.lower().str.replace(" ", "_")

# obtain numerical and categorical features
df['education'] = df['education'].astype('object')
cat_f = list(df.select_dtypes(include='object'))
num_f = list(df.drop(columns=cat_f).columns)
cat_f.remove('personality')

# encoding target variable
le = LabelEncoder()
df.personality = le.fit_transform(df.personality)

cat_f = [x for x in cat_f if x != 'gender']
num_f = [x for x in num_f if x not in ['age', 'judging_score']]

# aggregate used features
features = cat_f + num_f
features.append('personality')

# split data to train and test sets
df_full_train, df_test = train_test_split(df[features],
                                          test_size=0.2,
                                          random_state=random_state)
df_kfold = df_full_train.copy()

for _ in [df_full_train, df_test, df_kfold]:
    _.reset_index(drop=True)

# obtain features and targets
def getXy(df, target):
    return df.pop(target), df

# get features and target
y_full_train, X_full_train = getXy(df_full_train, 'personality')
y_test, X_test = getXy(df_test, 'personality')

# validate model with KFold cross-validation
print("Validating using KFold cross-validation...")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_kfold):
    target = 'personality'

    y_train, X_train = getXy(df_kfold.iloc[train_idx], target)
    y_val, X_val= getXy(df_kfold.iloc[val_idx], target)

    testmodel = TestModel(X_train, y_train,
                          X_val, y_val,
                          num_f, cat_f)

    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=random_state,
                                         learning_rate=0.1, max_depth=3, min_samples_leaf=3)
    score = testmodel.get_result(hgb)
    scores.append(score)
    print(f'auc on fold {fold} is {score}')
    fold = fold + 1

# train final model
print('Train final model...')
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=random_state,
                                     learning_rate=0.1, max_depth=3, min_samples_leaf=3)

final_model = TestModel(X_full_train, y_full_train,
                          X_test, y_test,
                          num_f, cat_f)

final_model.get_result(hgb)

# export model
with open(output_file, 'wb') as f_out:
    pickle.dump((final_model, hgb), f_out)

print(f'Export model at {output_file}')
