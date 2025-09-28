import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
veriler = pd.read_csv("missing_data_example.csv")
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
salary = veriler.iloc[:,1:4].values
imputer = imputer.fit(salary[:,1:4])
salary[:,1:4]=imputer.transform(salary[:,1:4])
print(salary) # bu şekilde eksik verileri tamamlamış olduk





#rint(veriler)