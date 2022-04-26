import numpy as np
import numexpr as ne
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
import pingouin as pg
import scipy as sp
from icecream import ic

guess_vd = [19.26434602795052, 0.32669220692084294, -0.0005527079074078076, -0.1212851299830873, 0.00038333912648868466, -27.130732939248595]
guess_wd = [-35.38159007494937, -0.35533831622314166, 0.0008836088655352115, 0.18673429297445848, -0.0006246725944855244, 21.305037457616095]

header = ['A','Z','N','isospin','E','Betas','Chi','Mid Chi','V_d','W_d','V_d_orig','W_d_orig','Mag_orig']
df = pd.read_csv('results_original.txt', delim_whitespace = True, usecols = [2,3,4,5,6,7,8,9,10,11,12,13,15],skipinitialspace = True,skiprows=2,skipfooter=2,engine ='python',names = header)

fact = 3.5

sum_chi=df['Chi'].sum()
length = df['A'].shape[0]-1
std_deviation = sum_chi/length*fact 
df.sort_values('Chi', inplace=True)
spot = df[df['Chi']>std_deviation].index.size 
print('High Chi:') 
print(df.tail(spot).to_string()) 
df = df.head(length-spot) 
dfObj_neg_values = pd.DataFrame(columns = header)
df.sort_values(['A','E','isospin'], inplace=True)
true_A = df['A']
true_E = df['E']
true_V = df['V_d_orig']
true_W = df['W_d_orig']
true_B = df['Betas']

dfObj_neg_values = pd.concat([dfObj_neg_values, df[df['V_d'] < df['V_d_orig']]])
dfObj_neg_values = pd.concat([dfObj_neg_values, (df[((df['W_d'] < 0.0) & (df['W_d_orig']> 0.0))])]) 
dfObj_neg_values = pd.concat([dfObj_neg_values, df[df['W_d'] > df['W_d_orig']]])
#add higher chi values to mix
dfObj_neg_values = pd.concat([dfObj_neg_values, df[df['Mid Chi'] < df['Chi']]]) 
pd.set_option('display.max_rows', 1000) 
print('unphysical points thrown out') 
print(dfObj_neg_values.to_string()) 
ic('No of high Chi-Sqr dropped, ',fact,' above average ' + str(spot)) 
ic('No of unphysical points dropped: '+str(dfObj_neg_values.shape[0])) 
df = df.drop(df[df['V_d'] < df['V_d_orig']].index) 
df = df.drop(df[((df['W_d'] < 0.0) & (df['W_d_orig']> 0.0))].index) 
df = df.drop(df[df['W_d'] > df['W_d_orig']].index)
df = df.drop(df[df['Mid Chi'] < df['Chi']].index) 
ic('Percent remaining: '+str(df.shape[0]*100/length))
ic('Mid/Last Ratio: ' + str(df['Mid Chi'].sum()/df['Chi'].sum()))

df.loc[(df['isospin'] == 'n'), 'isospin'] = -0.5
df.loc[(df['isospin'] == 'p'), 'isospin'] = 0.5
df['isospin'] = df['isospin'].astype(float, errors = 'raise')

A = df['A']
original_A = df['A']
V_d = df['V_d']
W_d = df['W_d']
Betas = df['Betas']
original_Betas = df['Betas']
V_d_orig = df['V_d_orig']
original_V_d_orig = df['V_d_orig']
W_d_orig = df['W_d_orig']
original_W_d_orig = df['W_d_orig']
E = df['E']
original_E = df['E']


#n = df
#n.sort_values('Betas', inplace=True)
#n = n.tail(25)
#n['Betas'] = n['Betas'].apply(lambda x: x ** 0.25)
#print(n['Betas'].mean())
#print out cvs file
df.to_csv('results_original_trim.cvs', sep=' ')

A = df['A']
V_d = df['V_d']
Betas = df['Betas']
V_d_orig = df['V_d_orig']
E = df['E']
isospin = df['isospin']
df.to_csv('results_original_trim.txt', sep=' ')

def test(A,c0,c1,c2,c3,c4,c5):
    A_Betas = np.power(Betas,0.5)
    E_Betas = np.power(Betas,0.25)
    return (A_Betas*(c0+(c1*A)+(c2*(A*A)))) + V_d_orig + (E_Betas*((c3*E)+(c4*(E*E))))+ (Betas*(c5*V_d_orig))

def test_fn(A,c0,c1,c2,c3,c4,c5):
    A_Betas = np.power(Betas,0.5)
    E_Betas = np.power(Betas,0.25)
    return ne.evaluate('(A_Betas*(c0+(c1*A)+(c2*(A*A)))) + V_d_orig + (E_Betas*((c3*E)+(c4*(E*E))))+ (Betas*(c5*V_d_orig))')


popt, pcov = curve_fit(test, A, V_d,guess_vd)
df['V_d_fit'] = test_fn(A, *popt)
A = true_A
V_d_orig = true_V
E = true_E
Betas = true_B
V_d_fitted = test_fn(A, *popt)
V_d_orig = original_V_d_orig
A = original_A
E = original_E
Betas = original_Betas
df = df.assign(reducedv = (((df['V_d_fit'] - df['V_d']) ** 2)/(length-len(guess_vd))))
ic('Reduced Chi: '+ str(df['reducedv'].sum()))
dist_a = pearsonr(df['A'],df['V_d_fit'])[0]
dist_e = pearsonr(df['E'],df['V_d_fit'])[0]
dist_b = pearsonr(df['Betas'],df['V_d_fit'])[0]
dist_v = pearsonr(df['V_d_orig'],df['V_d_fit'])[0]

ic(dist_a,dist_e,dist_b,dist_v)
dist = np.sqrt(((dist_a ** 2)/(np.sqrt(4))) +((dist_e ** 2)/(np.sqrt(4)))+ ((dist_b ** 2)/(np.sqrt(4)))+((dist_v ** 2)/(np.sqrt(4))))
ic('Correlation', dist)

ic(popt.tolist())
v_d_popt = popt.tolist()

unique_A = sorted(np.unique(df['A'].tolist()))
unique_E = []
unique_B = []
unique_V = []
for elements in unique_A:
    sub = df.loc[df['A'] == int(elements)]
    unique_B.append(sub['Betas'].mean())
    unique_E.append(sub['E'].mean())
    unique_V.append(sub['V_d_orig'].mean())

plt.figure()
plt.plot(A.to_numpy(), V_d.to_numpy(), 'ko', label="Data")
E = unique_E
A = unique_A
Betas = unique_B
V_d_orig = unique_V
plt.plot(A, test_fn(A, *popt), 'r-', label="Curve fit")
E = original_E
A = original_A
Betas = original_Betas
V_d_orig = original_V_d_orig

def test(A,c0,c1,c2,c3,c4,c5):
    A_Betas = np.power(Betas,0.5)
    E_Betas = np.power(Betas,0.25)
    return (A_Betas*(c0+(c1*A)+(c2*(A*A)))) + W_d_orig + (E_Betas*((c3*E)+(c4*(E*E)))) + (Betas*(c5*W_d_orig))


def test_fn(A,c0,c1,c2,c3,c4,c5):
    A_Betas = np.power(Betas,0.5)
    E_Betas = np.power(Betas,0.25)
    return ne.evaluate('(A_Betas*(c0+(c1*A)+(c2*(A*A)))) + W_d_orig + (E_Betas*((c3*E)+(c4*(E*E)))) + (Betas*(c5*W_d_orig))')

popt, pcov = curve_fit(test, A, W_d,guess_wd)
df['W_d_fit'] = test_fn(A, *popt)
A = true_A
W_d_orig = true_W
E = true_E
Betas = true_B
W_d_fitted = test_fn(A, *popt)
W_d_orig = original_W_d_orig
A = original_A
E = original_E
Betas = original_Betas
df = df.assign(reducedw = (((df['W_d_fit'] - df['W_d']) ** 2)/(length-len(guess_wd))))
ic('Reduced Chi: '+ str(df['reducedw'].sum()))
dist_a = pearsonr(df['A'],df['W_d_fit'])[0]
dist_e = pearsonr(df['E'],df['W_d_fit'])[0]
dist_b = pearsonr(df['Betas'],df['W_d_fit'])[0]
dist_w = pearsonr(df['W_d_orig'],df['W_d_fit'])[0]

ic(dist_a,dist_e,dist_b,dist_w)
dist = np.sqrt(((dist_a ** 2)/(np.sqrt(4))) +((dist_e ** 2)/(np.sqrt(4)))+ ((dist_b ** 2)/(np.sqrt(4)))+((dist_w ** 2)/(np.sqrt(4))))
ic('Correlation', dist)
ic(popt.tolist())
w_d_popt = popt.tolist()
print(df.corr(method='spearman').to_string())
plt.figure()
unique_A = sorted(np.unique(df['A'].tolist()))
unique_E = []
unique_B = []
unique_W = []
for elements in unique_A:
    sub = df.loc[df['A'] == int(elements)]
    unique_B.append(sub['Betas'].mean())
    unique_E.append(sub['E'].mean())
    unique_W.append(sub['W_d_orig'].mean())

plt.plot(A.to_numpy(), W_d.to_numpy(), 'ko', label="Data")
E = unique_E
A = unique_A
Betas = unique_B
W_d_orig = unique_W
plt.plot(A, test_fn(A, *popt), 'r-', label="Curve fit")
E = original_E
A = original_A
Betas = original_Betas
W_d_orig = original_W_d_orig
plt.legend()
#plt.show()


with open(__file__, 'r') as f:
    lines = f.read().split('\n')
    vd = 'guess_vd = ' + str(v_d_popt)
    wd = 'guess_wd = ' + str(w_d_popt)
    lines[13] = vd
    lines[14] = wd

with open(__file__, 'w') as f:
    f.write('\n'.join(lines[0:]))

#print(true_A.tolist())
#print(true_E.tolist())
#print((V_d_fitted - true_V).tolist())
#print((W_d_fitted - true_W).tolist())
