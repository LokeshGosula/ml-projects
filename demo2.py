import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
d=np.asarray([[1,2],
         [2,3],
         [3,4]])
x=[10,20,30]
print(d)
trans=PolynomialFeatures(degree=3)

result=trans.fit_transform(d)
print(result)
model=LinearRegression()
r=model.fit(result,x)
print(r)








