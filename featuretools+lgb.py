#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import featuretools as ft


# In[2]:



test=pd.read_csv('招行testB.csv')


# In[3]:


train=pd.read_csv('招行training.csv')


# In[4]:


label=train['LABEL']
train=train.drop(['LABEL'],axis=1).copy()


# In[19]:


combi=pd.concat([train,test],axis=0)


# In[20]:


es=ft.EntitySet(id='bank')


# In[21]:


es.add_dataframe(dataframe_name='zhaohang',dataframe=combi,index='CUST_UID')


# In[22]:


trans_primitives=['subtract_numeric','divide_numeric']
#agg_primitives=['sum','median','mean']


# In[23]:


feature_matrix,feature_names=ft.dfs(entityset=es,
                                    target_dataframe_name='zhaohang',
                                    max_depth=1,
                                    verbose=1,
                                    #agg_primitives=agg_primitives,
                                    trans_primitives=trans_primitives,
                                    n_jobs=-1)
                                    


# In[15]:


feature_matrix.columns


# In[29]:


feature_matrix=feature_matrix.reset_index()


# In[30]:


feature_matrix=feature_matrix.drop('CUST_UID',axis=1).copy()


# In[34]:


train=feature_matrix[:40000]
test=feature_matrix[40000:]


# In[ ]:


select_feature=['CUR_MON_COR_DPS_MON_DAY_AVG_BAL - ICO_CUR_MON_ACM_TRX_AMT',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - REG_DT',
 'LAST_12_MON_COR_DPS_DAY_AVG_BAL - LGP_HLD_CARD_LVL',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - HLD_FGN_CCY_ACT_NBR',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - NB_CTC_HLD_IDV_AIO_CARD_SITU',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - CUR_YEAR_COUNTER_ENCASH_CNT',
 'LAST_12_MON_COR_DPS_DAY_AVG_BAL / CUR_MON_COR_DPS_MON_DAY_AVG_BAL',
 'CUR_YEAR_MON_AGV_TRX_CNT - MON_12_AGV_LVE_ACT_CNT',
 'MON_6_50_UP_LVE_ACT_CNT / CUR_MON_COR_DPS_MON_DAY_AVG_BAL',
 'HLD_DMS_CCY_ACT_NBR',
 'CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL - CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - WTHR_OPN_ONL_ICO',
 'LAST_12_MON_COR_DPS_DAY_AVG_BAL - NB_CTC_HLD_IDV_AIO_CARD_SITU',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - NB_RCT_3_MON_LGN_TMS_AGV',
 'LAST_12_MON_COR_DPS_DAY_AVG_BAL / LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
 'LAST_12_MON_MON_AVG_TRX_AMT_NAV - REG_DT',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL / AI_STAR_SCO',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - MON_12_TRX_AMT_MAX_AMT_PCTT',
 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL / REG_CPT',
 'ICO_CUR_MON_ACM_TRX_AMT / MON_12_EXT_SAM_AMT',
 'ICO_CUR_MON_ACM_TRX_AMT - MON_12_TRX_AMT_MAX_AMT_PCTT',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL - MON_6_50_UP_ENTR_ACT_CNT',
 'CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR / REG_CPT',
 'REG_CPT / MON_12_EXT_SAM_AMT',
 'CUR_YEAR_MON_AGV_TRX_CNT / LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
 'MON_12_CUST_CNT_PTY_ID / MON_12_TRX_AMT_MAX_AMT_PCTT',
 'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT - LAST_12_MON_COR_DPS_DAY_AVG_BAL',
 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL / MON_12_TRX_AMT_MAX_AMT_PCTT',
 'MON_12_TRX_AMT_MAX_AMT_PCTT / CUR_MON_COR_DPS_MON_DAY_AVG_BAL',
 'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY / NB_RCT_3_MON_LGN_TMS_AGV',
 'REG_CPT / LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
 'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY / REG_CPT',
 'REG_CPT / SHH_BCK',
 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL - NB_RCT_3_MON_LGN_TMS_AGV']


# In[114]:


trainset_select=train[select_feature]
test_select=test[select_feature]


# In[115]:


train_select_X,valid_select_X,train_select_y,valid_select_y=train_test_split(trainset_select,label,stratify=label)
lgb_train = lgb.Dataset(train_select_X, train_select_y)
params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc'
    }
lgb_eval = lgb.Dataset(valid_select_X, valid_select_y, reference=lgb_train)

gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=500,early_stopping_rounds=50)


# In[104]:


test_pred=gbm.predict(test_select, num_iteration=gbm.best_iteration)


# In[105]:


ID=pd.read_csv('招行testIDB.csv')
ID['prob']=test_pred


# In[ ]:




