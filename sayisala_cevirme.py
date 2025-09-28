import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
veriler = pd.read_csv("eksikveriler.csv")
ulke = veriler.iloc[:,0:1].values
print(ulke)
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1])
print(ulke) # burada yaptığımız ülke gibi string ifadeyi sayısal yaptık
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke) # sayısal verileri binary e çevirme işlemi grçklşti.