"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("income.csv")0
print(df)

plt.scatter(df.Age, df[['Income($)']])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

model = KMeans(n_clusters=3)
Y_predict = model.fit_predict(df[['Age', 'Income($)']])

print(model.cluster_centers_)
df['cluster'] = Y_predict
print(df)


df1=df[df.cluster==0]
print(df1)
df2=df[df.cluster==1]
df3=df[df.cluster==2]



plt.scatter(df1['Age'], df1['Income($)'],color='red')
plt.scatter(df2['Age'], df2['Income($)'],color='black')
plt.scatter(df3['Age'], df3['Income($)'],color='blue')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],color='green',marker='*',label="centroid")
plt.show()



scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.Age)


scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
print(df)

df1=df[df.cluster==0]
print(df1)
df2=df[df.cluster==1]
df3=df[df.cluster==2]


SSE=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df[['Age','Income($)']])
    SSE.append(kmeans.inertia_)
    print(kmeans.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(range(1,10), SSE)
plt.show()
"""

import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

df=pd.read_csv('income.csv')
print(df)
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.Age)
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
print(df)
model=KMeans(n_clusters=3)
y_predict = model.fit_predict(df[["Age","Income($)"]])
print(model)
print(model.cluster_centers_)
print(model.inertia_)
print(y_predict)
df['cluster']=y_predict
print(df)

df1=df[df.cluster==0]
print(df1)
df2=df[df.cluster==1]
df3=df[df.cluster==2]
print(df2)
print(df3)
plt.scatter(df1['Age'], df1['Income($)'],color='red')
plt.scatter(df2['Age'], df2['Income($)'],color='black')
plt.scatter(df3['Age'], df3['Income($)'],color='blue')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color='green',marker="*",label="centroid")
plt.show()

print(model.inertia_)
sse=[]
lr=range(1,10)
for i in lr:
    km=KMeans(n_clusters=i)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.plot(lr,sse,color='blue')
plt.show()



