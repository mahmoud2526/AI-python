from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree

file=pd.read_csv("loan_data.csv")

filed=pd.get_dummies(file,columns=["purpose"],drop_first=False)


x=filed.drop("not.fully.paid",axis=1)

y=filed["not.fully.paid"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=70)

train=GaussianNB()
tree=DecisionTreeClassifier()
errors=[]
for i in range(1,10):
    k_neight=KNeighborsClassifier(n_neighbors=i)
    k_neight.fit(xtrain,ytrain)
    k_predict=k_neight.predict(xtest)
    acc3=accuracy_score(k_predict,ytest)
    errors.append(acc3)
train.fit(xtrain,ytrain)
tree.fit(xtrain,ytrain)

predict=train.predict(xtest)
tree_predict=tree.predict(xtest)
acc=accuracy_score(predict,ytest)
acc2=accuracy_score(tree_predict,ytest)

print(acc)
print(f"accuracy of deciation tree :{acc2}")
print(errors)
