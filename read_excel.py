import pandas as pd

# data=pd.read_excel("presidents_names.xlsx")
# print(data)
# c=data.loc[[1,2],["Name","born"]]
# print(c)
# v=data.iloc[[1,2],[2,3]]
# print(v)
with pd.ExcelFile("presidents_names.xlsx") as xl:
    f1=pd.read_excel(xl,"Sheet1")
    f2=pd.read_excel(xl,"Sheet2")
print(f1)
print(f2)
print(f2[0:3]["Name"])
