import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('salaries.csv')
print(df)
print(df.columns)
print(df.head())
lb=LabelEncoder()
df['company_n'] =lb.fit_transform(df['company'])
df['job_n'] =lb.fit_transform(df['job'])
df['degree_n'] =lb.fit_transform(df['degree'])
print(df.head(6))
target=df['salary_more_then_100k']
print(target)
df.drop(['salary_more_then_100k','company','job','degree'],axis=1,inplace=True)
print(df)
X=df
y=target
decision = DecisionTreeClassifier()
decision.fit(X,y)
print(decision.predict([[1,2,0]]))
