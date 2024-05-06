import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Melbourne_housing_FULL.csv")
print(df.head(10).values)
print(df.columns)

print(df.isna().sum())
print(df.shape)

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG',
'Regionname', 'Propertycount', 'Distance', 'CouncilArea',
'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df= df[cols_to_use]
print(df.head(4))
print(df.isna().sum())
fill0= ['Propertycount', 'Distance', 'Bedroom2',
'Bathroom', 'Car']

df[fill0]=df[fill0].fillna(0)
print(df.isna().sum())
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())
print(df.isna().sum())
df.dropna(inplace=True)
df=pd.get_dummies(df,drop_first=True)

print(df.head().values)
print(df.isna().sum())
print(df.shape)
X=df.drop('Price',axis=1)
y=df.Price


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))
print(lr.score(X_train,y_train))




from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 50, max_iter = 100, tol = 0.1)
ridge.fit(X_train, y_train)
print(ridge.score(X_test, y_test))
print(ridge.score(X_train,y_train))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=50,max_iter=100,tol=0.1)
lasso.fit(X_train, y_train)
print(lasso.score(X_test, y_test))
print(lasso.score(X_train,y_train))
