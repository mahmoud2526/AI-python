from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC

dig=load_digits()

digital1=dig.images[450]
plt.imshow(digital1,cmap=plt.cm.binary)
print(dig.target[12])
xtrain,xtest,ytrain,ytest=train_test_split(dig.data,dig.target,test_size=.33)

svc=SVC(kernel="poly",degree=2,gamma=2,C=1)
svc.fit(xtrain,ytrain)

score=svc.score(xtest,ytest)
print(svc.get_params())

# params={
#     "C":[0,1,2,5,10,20,50,100],
#     "gamma":[2,4,6,8,10],
#     "degree":[2,3,5,7,9,10],
#     "kernel":["linear","rbf","poly"]
#
#ensassble classifier
#XG boost
#bagging and bosting
# }
#
#
# grid=GridSearchCV(svc,params)
# grid.fit(xtrain,ytrain)
# print(grid.best_params_)

print(score)
plt.show()
