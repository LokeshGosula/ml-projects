from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
import keras
warnings.filterwarnings('ignore')
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ',')
# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]
# define the keras model
model = Sequential()
model.add(Dense(units=8, activation = 'relu'))
model.add(Dense(units=12, activation = 'relu'))
model.add(Dense(units=1, activation = 'sigmoid'))
print("Model created")
model.compile(loss="binary_crossentropy",optimizer='Adam', metrics=['accuracy'])
model.fit(X,y,epochs=100,batch_size=10, verbose=0)
df=model.get_weights()
print(df)
for values in df:
    print(values)
