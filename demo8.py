import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
df['target']=iris.target
print(df)
df["flower_names"]=df.target.apply(lambda x: iris.target_names[x])
print(df)

X = df.drop(['target','flower_names'],axis=1).values
y = df.target.values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=3)

model=RandomForestClassifier(n_estimators=40)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(X_test)
print(model.predict(X_test))
print(model.predict([[5.5,3.5,1.3,0.2]]))


