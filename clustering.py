import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.cluster import AgglomerativeClustering,DBSCAN,MeanShift
file=pd.read_csv("Live_20210128.csv")
file.drop(["Column1","Column2","Column3","Column4","status_published","status_id"],axis=1,inplace=True)



label=LabelEncoder()
file["status_type"]=label.fit_transform(file["status_type"])
c=file.columns
minmax=MinMaxScaler()
file=minmax.fit_transform(file)

data=pd.DataFrame(file,columns=c)

agg=AgglomerativeClustering(n_clusters=3)
agg.fit(data)

DBsacn=DBSCAN(eps=4,min_samples=6)
DBsacn.fit(data)

mean=MeanShift(bandwidth=5)
mean.fit(data)


print(data)

