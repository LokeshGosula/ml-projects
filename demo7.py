import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

iris=load_iris()
print(dir(iris))
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

df['target']=iris.target
a=lambda x:iris.target_names[x]
df['target_names']=df.target.apply(a)

print(df.head())

X=df.drop(['target','target_names'],axis='columns')
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y)

rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
print(rf.score(X_test,y_test))
print(y_pred)

print(rf.score(X_train,y_train))
print(rf.predict([[5.1,3.5,1.4,0.2]]))
