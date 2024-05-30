import pandas as pd
import sklearn.preprocessing as skp
data=pd.read_csv("pima-indians-diabetes.csv")
# print(data)
data1=skp.MinMaxScaler(feature_range=(0,1))
print(data1)
print(data)
# d=data.iloc[:,[2,5]]
# print(d)
# data2=data1.fit_transform(d)
# print(data2)
