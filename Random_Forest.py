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

from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

'''from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
'''

# karar ağacı uygulaması

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X +0.5
K = Y-0.4
plt.scatter(X,Y, color="red")
plt.plot(x,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y,color="green")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.plot(X,rf_reg.predict(X),color="red")
# karar agacı olan değerleri bulurken randomforest daha ortalama bir değer buluyor bu uygulamada daha avantajlı
plt.plot(X,rf_reg.predict(Z),color="red")
plt.plot(X,rf_reg.predict(K),color="yellow")
# burada ise daha ortalama bir grafik oluştuğunu görebiliyoruz

 

