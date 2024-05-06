import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
digits = load_digits()
print(len(digits.data))
print(digits.images)
print(dir(digits))
print(digits.DESCR)
print(digits.images[0])
'''plt.matshow(digits.images[0])
plt.gray()
plt.show()
plt.matshow(digits.images[2])
plt.gray()
plt.show()
'''
print()
print()
print(digits.data)
print()
print()
print(digits.target)
X= digits.data
y= digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=3,random_state=10)
print(X_train)
print(X_test)
result = LogisticRegression(solver='lbfgs',max_iter=5000)
result.fit(X_train,y_train)
print(result.predict(X_test))