#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cbt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn import metrics


# In[3]:


train=pd.read_csv('招行training.csv')
test=pd.read_csv('招行test.csv')
#train_xgb=pd.read_csv('招行training_one_hot.csv')
#test_xgb=pd.read_csv('招行test_one_hot.csv')                      
ID=pd.read_csv('招行testID.csv')
#train1=pd.read_csv('招行training.csv')


# In[4]:


lgbmodel=lgb.LGBMClassifier(
 boosting_type='gbdt',
 metrics='auc',
 objective='binary',
 learning_rate=0.1,
 n_estimators=74, 
 max_depth=7, 
 num_leaves=50,
 max_bin=165,
 min_data_in_leaf=51,
 bagging_fraction=0.6,
 bagging_freq= 0, 
 feature_fraction= 0.8,
 lambda_l1=1e-05,
 lambda_l2=1e-05,
 min_split_gain=0,
 random_state=32)

xgbmodel=xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=5,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=32)

catmodel=cbt.CatBoostClassifier(
 iterations=1000, 
 learning_rate=0.1,
 depth=8,
 random_seed=32)


# In[5]:



#train_X=train.drop('LABEL',axis=1).copy()
#train_y=train['LABEL'].copy()
X_train=train.drop('LABEL',axis=1).copy()
y_train=train['LABEL'].copy()
X_test=test
#for c in ['MON_12_CUST_CNT_PTY_ID','WTHR_OPN_ONL_ICO','LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU']:
   # train1_X[c]= train1_X[c].astype('category').cat.codes
#X_train1,X_test1,y_train1,y_test1=train_test_split(train1_X,train1_y,random_state=42,stratify=train1_y)

#category_feature=['MON_12_CUST_CNT_PTY_ID','WTHR_OPN_ONL_ICO','LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU']


# In[6]:


oof_lgb = np.zeros(X_train.shape[0])
oof_xgb = np.zeros(X_train.shape[0])
oof_cat = np.zeros(X_train.shape[0])

test_output = pd.DataFrame(columns=['lgb','xgb','cat'],index=range(X_test.shape[0]))
test_output = test_output.fillna(0)

kfold = KFold(n_splits=5)


# In[6]:


for train_idx, valid_idx in kfold.split(X_train):
    train_x = X_train.iloc[train_idx]
    train_y = y_train.iloc[train_idx]
    
    valid_x = X_train.iloc[valid_idx]
    valid_y = y_train.iloc[valid_idx]
    
    lgbmodel.fit(train_x,train_y, eval_set=[(train_x,train_y),(valid_x,valid_y)],early_stopping_rounds=50,
                 eval_metric='auc')
    xgbmodel.fit(train_x,train_y, eval_set=[(train_x,train_y),(valid_x,valid_y)],early_stopping_rounds=50,eval_metric='auc')
    catmodel.fit(train_x,train_y, eval_set=[(train_x,train_y),(valid_x,valid_y)],early_stopping_rounds=50,
                 )
    
    oof_lgb[valid_idx] = lgbmodel.predict_proba(valid_x)[:,1]
    oof_xgb[valid_idx] = xgbmodel.predict_proba(valid_x)[:,1]
    oof_cat[valid_idx] = catmodel.predict_proba(valid_x)[:,1]
    test_output['lgb'] += lgbmodel.predict_proba(X_test)[:,1]
    test_output['xgb'] += xgbmodel.predict_proba(X_test)[:,1]
    test_output['cat'] += catmodel.predict_proba(X_test)[:,1]

test_output['lgb'] = test_output['lgb'] / 5
test_output['xgb'] = test_output['xgb'] / 5
test_output['cat'] = test_output['cat'] / 5


# In[7]:


oof_df = pd.DataFrame({'lgb':oof_lgb,'xgb':oof_xgb,'cat':oof_cat})
#oof_df1 = pd.DataFrame({'lgb':oof_lgb1,'xgb':oof_xgb1,'cat':oof_cat1})


# In[10]:


ID['prob']=final_pre[:,1]


# In[11]:


ID.to_csv('D:\CUHKSZ第二学期\招行fintech'+ 'test_A.txt', sep='\t',index=False, header = False)


# In[ ]:




