#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TestModel:
    def __init__(self, f_train, t_train, f_test, t_test, num_f, cat_f):
        self.X_train = f_train
        self.y_train = t_train
        self.X_test = f_test
        self.y_test = t_test
        self.num_f = num_f
        self.cat_f = cat_f
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def get_result(self, model):
        self.__f_fit()
        f_train = self.__f_transform(self.X_train)
        f_test = self.__f_transform(self.X_test)
        
        model.fit(f_train, self.y_train)
        y_pred = model.predict_proba(f_test)
        
        return roc_auc_score(self.y_test, y_pred, multi_class='ovr', average='weighted')
        
    def predict_proba(self, X_input, model):
        X = self.__f_transform(X_input)
        return model.predict_proba(X)

    def predict_class(self, X_input, model):
        X = self.__f_transform(X_input)
        return model.predict(X)
        
    def __f_fit(self):
        if self.num_f: self.scaler.fit(self.X_train[self.num_f].values)
        if self.cat_f: self.encoder.fit(self.X_train[self.cat_f].values)

    def __f_transform(self, X_input):
        X_num = self.scaler.transform(X_input[self.num_f].values)
        X_cat = self.encoder.transform(X_input[self.cat_f].values)
        return np.column_stack([X_num, X_cat])
