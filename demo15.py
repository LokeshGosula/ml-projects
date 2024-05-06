import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets,svm

iris = datasets.load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df)

df['target_names']=df.target.apply(lambda x:iris.target_names[x])
print(df.head())
X=df.drop(['target','target_names'],axis=1)
y=df['target']

d={
    'c':[1,10,20],
    "kernel":["linear","poly","rbf"]
}
gs=GridSearchCV(
    svm.SVC(gamma='auto'),d,cv=5,return_train_score=False
)
gs.fit(iris.data,iris.target)
print(gs.best_params_)