import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC

file=pd.read_csv("diabetes.csv")
c=file.columns

#file imputer and scaled
# imputer=SimpleImputer(missing_values=0,strategy='mean')
scaled=MinMaxScaler(feature_range=(0,1))
scaled=scaled.fit_transform(file)       #scaled
#data frame
clear=pd.DataFrame(scaled,columns=c)  #data for scaled
# clear=imputer.fit_transform(im) #fit for missing
# datae=pd.DataFrame(imputerd,columns=c) #data frame for misssing
print(clear)
# print(datae.transpose())
# print("****************************************************************")
# print(datae.describe())

# x=datae.drop("Outcome",axis=1)
# y=datae["Outcome"]
x_metrics=clear.drop("Outcome",axis=1)
col=x_metrics.columns
imputer=SimpleImputer(missing_values=0,strategy='mean')
x_imputer=imputer.fit_transform(x_metrics)
x=pd.DataFrame(x_imputer,columns=col)
y=clear["Outcome"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=42)
tree=DecisionTreeClassifier()
tree.max_depth=4
tree.max_features=5
# tree.max_leaf_nodes=10
print(tree.get_params())
tree.fit(xtrain,ytrain)
tpredict=tree.predict(xtest)
error=accuracy_score(tpredict,ytest)
# print(datae)
print(error)

svc=SVC(kernel='rbf')
svc.fit(xtrain,ytrain)
svc_perdict=svc.predict(xtest)
acc=accuracy_score(svc_perdict,ytest)

plot=plot_tree(tree,feature_names=c)
print(acc)
# plt.imshow(plot)
plt.show()
