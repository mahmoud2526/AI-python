import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets  import load_iris
from sklearn.decomposition import PCA
from  sklearn.linear_model import  LogisticRegression
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file=load_iris()

x=file.data
y=file.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

values=[]
for i in range(1,x.shape[1]+1):
    pca=PCA(n_components=i)
    x_new_train=pca.fit_transform(xtrain)
    values.append(np.sum(pca.explained_variance_ratio_))

plt.plot(range(1,x.shape[1]+1),values,marker="o")

value=np.argmax(np.diff(values))+1
pca=PCA(n_components=value)
xtrain2=pca.fit_transform(xtrain)
xtest2=pca.fit_transform(xtest)

logic=LogisticRegression()
# logic.fit(xtrain2,ytrain)
#
# predict2=logic.predict((xtest2))

logic.fit(xtrain,ytrain)
predict=logic.predict(xtest)
# acc=accuracy_score(predict2,ytest)
acc2=accuracy_score(predict,ytest)
print(acc2)

plt.show()
