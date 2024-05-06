import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv('homeprices1.csv')
print(df)
df['bedrooms']=df['bedrooms'].fillna(df['bedrooms'].median())
print(df)
X=df.drop('price',axis=1).values
df1=df.area.values
y=df['price'].values
print(X)
print(y)
reg=LinearRegression()
result =reg.fit(X,y)
print(result.predict(X))
print(y)
print()
print()
print(result.predict([[4500,3,35]]))
plt.scatter(df1,y,color='red')


