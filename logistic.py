import numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer,load_digits
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=load_breast_cancer()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)
predict=model.predict(x_test)
classification_report=classification_report(predict,y_test)
print("classification_report",classification_report)
accuracy_score=accuracy_score(predict,y_test)
print("accuracy_score",accuracy_score)
confusion_matrix=confusion_matrix(predict,y_test)
print("confusion_matrix",confusion_matrix)

