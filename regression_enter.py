import pandas as pd
import numpy as np
import matplotlib as plt
veriler = pd.read_csv("tamamlanmis_satislar.csv")
aylar = veriler[["Aylar"]]
print(aylar)
satislar = veriler[["Satislar"]]
print(satislar)
satislar2 = veriler.iloc[:,:1].values
print(satislar2)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =sc.fit_transform(x_train) 
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train) # verilmiş olan bilgilere göre x trainden ytraini tahmin ediyor


