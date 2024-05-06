import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df= pd.read_csv('poly_dataset.csv')
print(df)
df1=df.Salary.values
X=df.Level.values.reshape(-1,1)
feature=PolynomialFeatures(degree=3)
y=feature.fit_transform(X)
print(df1)
print(y)
lr=LinearRegression()
model=lr.fit(y,df1)
print(model)




