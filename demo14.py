import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np


df=pd.read_csv('Titanic-Dataset.csv',usecols = ['Pclass', 'Sex', 'Age', 'Fare',
'Survived'])
print(df.head())
print(df.columns)

print(df.isna().sum())
df.Age = df.Age.fillna(df.Age.median())
print(df.isna().sum())

X=df.drop(['Survived','Sex'],axis=1)
print(X.head())
dummies = pd.get_dummies(df.Sex,dtype=int)
df1=pd.concat([df,dummies],axis='columns')
print(df1.head())
y=df['Survived']

nb=GaussianNB()
nb.fit(X,y)
print(nb.score(X,y))
print(np.average(cross_val_score(nb,X,y,cv=5)))
