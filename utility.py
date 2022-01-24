
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
get_ipython().system('pip install catboost')
from tqdm import tqdm 
import numpy as np
from catboost import CatBoostClassifier
import os
def setup():
  
  os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/My Drive/Kaggle"
  get_ipython().run_line_magic('cd', '/content/drive/My Drive/Kaggle')
  get_ipython().system('kaggle competitions download -c instacart-market-basket-analysis')
  get_ipython().system("unzip orders.csv.zip -d '/content/'")
  get_ipython().system("unzip order_products__train.csv.zip -d '/content/'")
  get_ipython().run_line_magic('cd', '/content/')
  preprocessed_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Untitled folder/data/preprocessed.csv')
  
  orders = pd.read_csv('/content/orders.csv')
  temp=orders[orders['eval_set']=='train'][['order_id','user_id','order_dow','order_hour_of_day','days_since_prior_order']]  
  preprocessed_data=pd.merge(preprocessed_data,temp,on='user_id',how='inner')
  train_data = pd.read_csv('/content/order_products__train.csv')
  temp2=pd.merge(orders,train_data,on='order_id',how='inner')[['user_id','product_id','reordered']]
  preprocessed_data=pd.merge(preprocessed_data,temp2,on=['user_id','product_id'],how='left')
  for i in range(len(preprocessed_data)):
    if str(preprocessed_data['reordered'][i])=='nan':
      preprocessed_data['reordered'].values[i]=0

  pca=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Untitled folder/data/pca_feature.csv')
  preprocessed_data=pd.merge(preprocessed_data,pca,on=['user_id'],how='inner')
  filename = '/content/drive/MyDrive/Colab Notebooks/Untitled folder/finalized_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  return preprocessed_data,loaded_model


# In[ ]:


"""
@author: Faron
"""

'''
This kernel implements the O(n²) F1-Score expectation maximization algorithm presented in
"Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012."

It solves argmax_(0 <= k <= n,[[None]]) E[F1(P,k,[[None]])]
with [[None]] being the indicator for predicting label "None"
given posteriors P = [p_1, p_2, ... , p_n], where p_1 > p_2 > ... > p_n
under label independence assumption by means of dynamic programming in O(n²).
'''


class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    def __init__(self):
        pass
    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone
def print_best_prediction(product,P, pNone=None):
    #print("Maximize F1-Expectation")
    P = np.sort(P)[::-1]
    n = P.shape[0]
    #L = ['L{}'.format(i + 1) for i in range(n)]

    if pNone is None:
        #print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
        pNone = (1.0 - P).prod()

    #PL = ['p({}|x)={}'.format(l, p) for l, p in zip(L, P)]
    #print("Posteriors: {} (n={})".format(PL, n))
    #print("p(None|x)={}".format(pNone))

    opt = F1Optimizer.maximize_expectation(P, pNone)
    best_prediction = ['None'] if opt[1] else []
    best_prediction += (product[:opt[0]])
    return best_prediction
    #print("Prediction {} yields best E[F1]n".format(best_prediction))


# In[ ]:


def max_f1_output(test_data,preds):
  temp=pd.DataFrame()
  temp['user_id']=test_data['user_id']
  temp['product_id']=test_data['product_id']
  temp['pred_reorder'] = preds
  temp=temp.sort_values(by=['user_id','pred_reorder'], ascending=False,)
  t1=temp.groupby('user_id')['product_id'].apply(list).reset_index(name='product_ids')
  t2=temp.groupby('user_id')['pred_reorder'].apply(list).reset_index(name='proba')
  temp=pd.merge(t1,t2,on='user_id',how='inner')
  preds=[]
  for idx,row in temp.iterrows():
    preds.append(print_best_prediction(row['product_ids'],row['proba']))
  temp['products']=preds
  temp=temp.drop(['product_ids','proba'], axis=1)
  orders = pd.read_csv( '/content/orders.csv')
  test_order=orders[orders['eval_set']=='train']
  test_order=test_order[['order_id','user_id']]
  test_order=pd.merge(test_order,temp,on='user_id',how='inner')
  test_order=test_order.drop(['user_id'], axis=1)
  return test_order


# In[ ]:


def predict_score(X_data,preds):
    pred=max_f1_output(X_data,preds)
    tp=0
    fn=0
    fp=0
    f1_score=[]
    order_train=pd.read_csv( '/content/order_products__train.csv')
    order_train=order_train[order_train['reordered']==1]
    temp=pd.DataFrame(order_train.groupby('order_id')['product_id'].apply(list).reset_index(name='true_products'))
    orders=pd.read_csv( '/content/orders.csv')
    temp=pd.merge(orders,temp,on='order_id',how='left')[['order_id','true_products']]
    for i in range(len(temp)):
      if str(temp['true_products'][i])=='nan':
        temp['true_products'].values[i]=['None']
  
    temp=pd.merge(pred,temp,on='order_id',how='inner')
    temp['true_products']=temp['true_products'].apply(set)
    temp['products']=temp['products'].apply(set)

    for j, row in temp.iterrows():
      tp=len(row['products'].intersection(row['true_products']))
      pression=tp/len(row['products'])
      recall=tp/len(row['true_products'])
      if tp==0:
        f1 = 0 
      else:
        f1=2*pression*recall/(pression+recall)
      f1_score.append(f1)
    mean_f1=sum(f1_score)/len(f1_score)
    return mean_f1
  


# In[ ]:


def predict_output(X_data,preds):
    pred=max_f1_output(X_data,preds)
    str_pred=[]
    for idx,row in pred.iterrows():
      s = [str(i) for i in row['products']]
      str_pred.append(" ".join(s))
    pred['products']=str_pred
    return pred

