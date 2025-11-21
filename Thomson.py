# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 22:10:20 2025

@author: ferha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000 # 10.000 tıklama
d = 10 # toplam 10 ilan var
tiklamalar = [0]* d # o ana kadarki tıklamalar

oduller = [0] * d # ilk başta bütün ilanların değeri 0 

toplam = 0 # toplam odul
secilenler = []
birler = [0]*d
sifirlar = [0]*d
for n in range(1,N):
    ad = 0 # secilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate ( birler[i]+ 1, sifirlar[i]+1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
            
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır bir ise 
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad] +1
        toplam = toplam + odul
print("Toplam Odul :")
print(toplam) 
plt.hist(secilenler)
plt.show()
          