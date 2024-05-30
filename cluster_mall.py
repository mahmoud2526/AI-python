import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.cluster import AgglomerativeClustering,DBSCAN,MeanShift


file=pd.read_csv("Mall_Customers.csv")
file.drop(["CustomerID"],axis=1,inplace=True)

#co varience matrix
label=LabelEncoder()
file["Gender"]=label.fit_transform(file["Gender"])
c=file.columns
minmax=MinMaxScaler()
file=minmax.fit_transform(file)

data=pd.DataFrame(file,columns=c)
print(data)

agg=AgglomerativeClustering(n_clusters=4)
agg.fit(data)

DBsacn=DBSCAN(eps=4,min_samples=3)
DBsacn.fit(data)

mean=MeanShift(bandwidth=4)
mean.fit(data)


print(agg.labels_)
print(DBsacn.labels_)
print(mean.labels_)
