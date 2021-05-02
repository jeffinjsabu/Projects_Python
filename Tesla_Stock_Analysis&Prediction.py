import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')



data=pd.read_csv("G:\Tesla_Stock_Analysis\TSLA.csv")

data.head()


data.shape


plt.figure(figsize=(16,8))
plt.title("Tesla Stock Analysis")
plt.xlabel('Date')
plt.ylabel("Close Prize in Dollars")
plt.plot(data["Close"])
plt.show()

data=data[["Close"]]
data.head()


future=25
data["Prediction"]=data[["Close"]].shift(-future)
data.tail()

x=np.array(data.drop(["Prediction"],1))[:-future]
print(x)

y=np.array(data["Prediction"])[:-future]
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

tree=DecisionTreeRegressor().fit(x_train,y_train)
lr=LinearRegression().fit(x_train,y_train)


x_future=data.drop(["Prediction"],1)[:-future]
x_future=x_future.tail(future)
x_future=np.array(x_future)
x_future


tree_prediction=tree.predict(x_future)
print(tree_prediction)
print()
lr_prediction=lr.predict(x_future)
print(lr_prediction)

#Final Prediction

Predictions=lr_prediction
valid=data[x.shape[0]:]
valid['Predictions']=Predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Closing price in USD')
plt.plot(data['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Predict'])
plt.show()
