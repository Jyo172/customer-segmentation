# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# code starts here
df = pd.read_csv(path)
df=df.dropna(axis=0,how='any')
#print(df['InvoiceNo'].str.contains('C'))
df=df[df['Country']=='United Kingdom']
df['Return']=df['InvoiceNo'].str.contains('C')
df['Purchase']=[0 if x == True else 1 for x in df['Return']]
print(df['Purchase'].value_counts(),df.shape)
#print(df.isnull().sum())
# code ends here


# --------------
# code starts here
from datetime import timedelta
customers=pd.DataFrame(df.CustomerID.unique(),dtype=int)
customers.columns=['CustomerID']
df['Recency']=pd.to_datetime("2011-12-10")-pd.to_datetime(df['InvoiceDate'])
df['Recency']=df.Recency.dt.days
temp=df[df['Purchase']==1]
recency=temp.groupby(by='CustomerID',as_index=False).min()
customers=customers.merge(recency[['CustomerID','Recency']])
print(customers.head())
# code ends here


# --------------
# code stars here
temp_1=df[['CustomerID','InvoiceNo','Purchase']]
temp_1.drop_duplicates(subset=['InvoiceNo'],inplace=True)
annual_invoice=temp_1.groupby(by='CustomerID',as_index=False).sum()
annual_invoice.rename(columns={'Purchase':'Frequency'},inplace=True)
customers=customers.merge(annual_invoice,on='CustomerID')
print(customers.head(),annual_invoice.head())
# code ends here


# --------------
# code starts here
df['Amount']=df.Quantity*df.UnitPrice
annual_sales=df.groupby(by='CustomerID',as_index=False).sum()
annual_sales.rename(columns={'Amount':'monetary'},inplace=True)
customers=customers.merge(annual_sales[['CustomerID','monetary']],on='CustomerID')
print(customers.head())
# code ends here


# --------------
# code ends here
customers['monetary']=np.where(customers['monetary']<0,0,customers['monetary'])
customers['Recency_log']=np.log(customers['Recency']+0.1)
customers['Frequency_log']=np.log(customers.Frequency)
customers['Monetary_log']=np.log(customers.monetary)
print(customers.head())
# code ends here


# --------------
# import packages
from sklearn.cluster import KMeans


# code starts here
dist=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(customers)
    dist.append(km.inertia_)
fig=plt.figure(figsize=[10,10])
plt.plot(range(1,10),dist)

# code ends here


# --------------
# code starts here
cluster=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
customers['cluster']=cluster.fit_predict(customers.iloc[:,1:7])
customers.plot.scatter(x='Frequency_log',y='Monetary_log',c='cluster',colormap='viridis')

# code ends here


