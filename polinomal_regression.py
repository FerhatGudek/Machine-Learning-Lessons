import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("unvan_maas.csv")
print(veriler)

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X, Y,color="blue")
plt.plot(x,lin_reg.predict(X),color="yellow")
# lineer bir grafik oluşturdu

# polynominal regression için yapacak olursak (2.dereceden)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
# burada ilk x değerini poliye çevirip oluşan değerleri tahmin ettirmeye çalışıyoruz
plt.scatter(X,Y,color="green")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="red")
plt.show()         

poly_reg = PolynomialFeatures(degree=4)
x_poly3 = poly_reg.fit_transform(X)
print(x_poly3)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)
# burada ilk x değerini poliye çevirip oluşan değerleri tahmin ettirmeye çalışıyoruz
plt.scatter(X,Y,color="green")
plt.plot(X,lin_reg3.predict(poly_reg.fit_transform(X)),color="red")
plt.show()         
# daha düzgün bir grafik oluşturdu 4. dereceden polinom grafiği

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg3.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg.fit_transform([[11]])))