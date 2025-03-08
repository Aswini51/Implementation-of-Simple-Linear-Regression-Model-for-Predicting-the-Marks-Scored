# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Developed by: G ASWINI
RegisterNumber:  24000247

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```

## Output:

![Screenshot 2025-03-08 142826](https://github.com/user-attachments/assets/1a09a45d-0634-4dbe-8a8c-e9d0ed07171c)

## Program:

dataset.info()

## Output:

![Screenshot 2025-03-08 142835](https://github.com/user-attachments/assets/75cb5959-a25e-48d9-9b2e-bd59b50c8c69)

## Program:

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

## Output:

![Screenshot 2025-03-08 142851](https://github.com/user-attachments/assets/0965ff57-4dc6-4913-a3cf-7487149265ba)

## Program:

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape

## Output:

![Screenshot 2025-03-08 142900](https://github.com/user-attachments/assets/c204bef6-2c54-4ced-b91b-c35fe9bfd3cd)

## Program:

X_test.shape

## Output:

![Screenshot 2025-03-08 142906](https://github.com/user-attachments/assets/089a0ceb-e1e2-43a9-a8f4-22f27c6b9d78)

## Program:

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

## Output:

![Screenshot 2025-03-08 142920](https://github.com/user-attachments/assets/1bc4f89e-def1-478c-995e-e79a05dbdec8)

## Program:

Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

## Output:

![Screenshot 2025-03-08 142927](https://github.com/user-attachments/assets/ac8ecb5d-55f2-45b0-94da-5f273c422556)

## Program:

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Training Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

## Output:

![Screenshot 2025-03-08 142938](https://github.com/user-attachments/assets/5f470666-115e-4aec-870b-b80192b6d376)

## Program:

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='silver')
plt.title('Test Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

## Output:

![Screenshot 2025-03-08 142953](https://github.com/user-attachments/assets/a09a55cb-80eb-4dce-834d-d82f03038399)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
