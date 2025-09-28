import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
veriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
yas= veriler.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)
#print(veriler)
#yapılan işlemler !!
"""
 celikle veriyi okuduk ardından eksik veriyi dolduracak makineyi
olusturduk daha sonra tüm veri setini taraması için iloc kullandık
ardından o aralığı fit ettik ve transform ederek değerler yazıldı
"""