
# coding: utf-8

# In[2]:


from flask import Flask, redirect, request,url_for
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier

# In[4]:


def setup():
  preprocessed_data=pd.read_csv('sample_data/sample_data.csv')
  preprocessed_data=preprocessed_data.drop(['reordered'],axis=1)
  orders = pd.read_csv('sample_data/sample_orders.csv')
  temp=orders[['user_id']]  
  preprocessed_data=pd.merge(preprocessed_data,temp,on='user_id',how='inner')
  train_data = pd.read_csv('sample_data/sample_order_products__train.csv')
  temp2=pd.merge(orders,train_data,on='order_id',how='inner')[['user_id','product_id','reordered']]
  preprocessed_data=pd.merge(preprocessed_data,temp2,on=['user_id','product_id'],how='left')
  for i in range(len(preprocessed_data)):
    if str(preprocessed_data['reordered'][i])=='nan':
      preprocessed_data['reordered'].values[i]=0

  filename = 'finalized_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  return preprocessed_data,loaded_model


# In[3]:


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


# In[5]:




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
  test_order = pd.read_csv( 'sample_data\sample_orders.csv')
  test_order=test_order[['order_id','user_id']]
  test_order=pd.merge(test_order,temp,on='user_id',how='inner')
  test_order=test_order.drop(['user_id'], axis=1)
  return test_order


# In[6]:


def predict_score(X_data,preds):
    pred=max_f1_output(X_data,preds)
    tp=0
    fn=0
    fp=0
    f1_score=[]
    order_train=pd.read_csv('sample_data\sample_order_products__train.csv')
    order_train=order_train[order_train['reordered']==1]
    temp=pd.DataFrame(order_train.groupby('order_id')['product_id'].apply(list).reset_index(name='true_products'))
    orders=pd.read_csv('sample_data\sample_orders.csv')
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
  


# In[7]:


def predict_output(X_data,preds):
    pred=max_f1_output(X_data,preds)
    str_pred=[]
    for idx,row in pred.iterrows():
      s = [str(i) for i in row['products']]
      str_pred.append(" ".join(s))
    pred['products']=str_pred
    return pred


# In[8]:


def pipeline_output(data):
  data=pd.merge(preprocessed_data,data,on=['order_id'])
  data=data.drop(['order_id'],axis=1)
  y_true=data['reordered']
  X=data.drop(['reordered'], axis=1)
  pred=loaded_model.predict_proba(X.iloc[:,2:])[:,1:]
  return predict_output(X,pred)


# In[9]:


import flask
preprocessed_data,loaded_model=setup()
app = Flask(__name__)


# In[ ]:


@app.route('/')
def hello_world():
    return redirect(url_for('index'))


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    data = list(set(str(to_predict_list['order_id']).split(',')))
    print(data)
    pred=pd.DataFrame()
    pred['order_id']=data
    pred['order_id']=pred['order_id'].astype(int)
    sub = pipeline_output(pred)
    html=sub.to_html()
    
    #write to html file
    text_file=open('output.html','w')
    text_file.write(html)
    text_file.close()
    return html
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

