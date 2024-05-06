import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
digits = load_digits()
print(dir(digits))
print(digits.target_names)
print(digits.data)
print(digits.target)
print(digits.feature_names)
X=digits.data
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
result1 = LogisticRegression()
result2=RandomForestClassifier()
result3=SVC()
score1 = cross_val_score(result1, X_train, y_train,cv=3)
print(np.average(score1))
score2 = cross_val_score(result2, X_train, y_train,cv=3)
score3 = cross_val_score(result3, X_train, y_train,cv=3)
print(np.average(score2))
print(np.average(score3))
