# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 23:15:23 2025

@author: ferha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000 # 10.000 tıklama
d = 10 # toplam 10 ilan var
tiklamalar = [0]* d # o ana kadarki tıklamalar

oduller = [0] * d # ilk başta bütün ilanların değeri 0 

toplam = 0 # toplam odul
secilenler = []
for n in range(1,N):
    ad = 0 # secilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: # max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad = i
            
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n,ad] # verilerdeki n. satır bir ise 
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul
    
print("Toplam Odul :")
print(toplam) 
plt.hist(secilenler)
plt.show()
          