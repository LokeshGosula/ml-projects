import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df =pd.read_csv('homeprices.csv')
print(df)

X=df.drop('price',axis=1).values
print(X)
y=df['price'].values
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

lr =LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
print(X_test)
print(y_pred)
print("x",lr.predict([[1400]]))

plt.plot(X_train,y_train)
