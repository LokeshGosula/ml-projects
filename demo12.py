import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))
print(iris.data)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
print(iris.target)
df['target']=iris.target
print(df.head())
df['flower_names']=df.target.apply(lambda x:iris.target_names[x])
print(df.head())
X=df.drop(['target','flower_names'],axis=1).values
print(X)
y=df.target.values
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=3)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print(y_test)
print()
print()
print(confusion_matrix(y_test, y_pred))

