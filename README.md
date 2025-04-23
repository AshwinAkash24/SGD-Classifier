# EX-07:SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
#### Step 1: Load the Data
We get flower data (Iris dataset) which has measurements and the type of flower.

#### Step 2: Split Data
We separate the data into:
X = the measurements
Y = the flower type (what we want to predict)

#### Step 3: Train-Test Split
We divide the data:
Training data to teach the model
Test data to check the model

#### Step 4: Train the Model
We use a model called SGDClassifier.
We teach it using the training data.

#### Step 5: Make Predictions
The model looks at the test data and guesses the flower type.

#### Step 6: Check Accuracy
We compare the model’s guesses with the real answers.
We check:
How many it got right (accuracy)
Where it was wrong (confusion matrix)

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
df
df.head()
df.tail()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
print(y_test)
print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:")
print(accuracy)
confusion_mat=confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(confusion_mat)
```

## Output:
![image](https://github.com/user-attachments/assets/32691a8d-cece-45dd-a552-fbbfdee15f1c)<br>
![image](https://github.com/user-attachments/assets/b9495d8a-9052-4f75-ab96-e27429306430)<br>
![image](https://github.com/user-attachments/assets/028a2f03-b8f7-49c0-8f56-20246b4acf96)<br>
![image](https://github.com/user-attachments/assets/e78d38f5-70a2-40d2-93a7-29eb6210f6a1)<br>
![image](https://github.com/user-attachments/assets/2fc9a69e-007e-4cea-bc15-74bfa7896b7e)<br>
![image](https://github.com/user-attachments/assets/5f6d0834-ca9e-44d3-9a27-27098a110342)<br>
![image](https://github.com/user-attachments/assets/ce59035c-9abc-4603-88e1-c743c165fb58)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
