import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
df['target']=iris.target
df['flowers_names']=df.target.apply(lambda x :iris.feature_names[x])

print(df)
X=df.drop(['target','flowers_names'],axis='columns')
print(X)
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3,random_state=42)
model= SVC(kernel='linear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)