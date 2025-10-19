import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
veriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
yas= veriler.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1])
print(ulke) # burada yaptığımız ülke gibi string ifadeyi sayısal yaptık
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke) # sayısal verileri binary e çevirme işlemi grçklşti.
c = veriler.iloc[:,-1:].values
print(c)
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c) # burada yaptığımız ülke gibi string ifadeyi sayısal yaptık
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c) # sayısal verileri binary e çevirme işlemi grçklşti
print(list(range(22)))
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc) # binary olan kısmın dataframi oluşturuldu
sonuc2 = pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2) # boy kilo yas verilerinin ismini belirmedik
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet) # cinsiyet değerlerini göstermiş olduk
sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])
print(sonuc3)
s = pd.concat([sonuc,sonuc2],axis=1)
print(s) # ülke ile boy kilo yas dframlerini birleştirdik
s2 = pd.concat([s,sonuc3],axis=1)
print(s2) # birlesmiş verileri cinsiyet ile birleştirdik
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)
