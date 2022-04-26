
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

header = ['A','Z','N','isospin','E','Betas','Chi','V_d','W_d','V_d_orig','W_d_orig']
df = pd.read_csv('results_final.txt', delim_whitespace = True, usecols = [2,3,4,5,6,7,8,10,11,12,13],skipinitialspace = True,skiprows=2,skipfooter=2,engine ='python',names = header)
df1 = pd.read_csv('results_original.txt', delim_whitespace = True, usecols = [2,3,4,5,6,7,8,10,11,12,13],skipinitialspace = True,skiprows=2,skipfooter=2,engine ='python',names = header)
df.sort_values(['A','E','isospin'], inplace=True)
df1.sort_values(['A','E','isospin'], inplace=True)

A = df['A'].tolist()
Chi_final = df['Chi'].tolist()
Chi_original = df1['Chi'].tolist()
total_chi = []

def formula(chi_f,chi_or):
    numerator = (chi_f-chi_or)*2
    denominator = chi_f+chi_or
    if denominator == 0.0:
        return 0.0
    else:
        return numerator/denominator

for i in range(len(A)):
    v = formula(Chi_final[i],Chi_original[i])
    if v >= 0.1 :
        print(df.iloc[[i]].to_string(index=False, header=False))
        print(df1.iloc[[i]].to_string(index=False, header=False))
        print('--------------------------------')
    total_chi.append(v)

plt.figure()
plt.plot(A, total_chi, 'ko', label="Data")
plt.legend()
plt.show()