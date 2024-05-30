import pandas as pd
import matplotlib.pyplot as plt

c=pd.read_csv("pandas exercise/Test.csv")
print(c)
print(c.dtypes)

d=c[c["Rating "]<2]
print(f"d",d)
