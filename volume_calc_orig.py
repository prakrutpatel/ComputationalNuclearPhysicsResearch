#   This program will try to recalculate a DWBA potential from  astandard elastic optical potential and recalculate it to work
# in a coupled channel code. It will read in the plot.param file. Inputs also include the distortion parameters
# Mix of Charlie and Steve

import numpy as np
from numpy import vectorize
import math
from scipy.integrate import tplquad, quad
from scipy import exp, pi, sin



#INPUTS:
# Is the Coulumb field distorted
coul_beta = True
# Do I use a global potential from Koenig,   Delaroche
glob=False

othird=1./3.


# This next section is awkward where the input data is stored in code. Better to have a series of input files
# beta is the deformation parameters
#Zn64
#glob=True
#E=20.4
#Z=30
#A=64
#no_betas = 4
#beta = np.zeros(no_betas)
#beta[0] = 0.25  # Deformation parameter, quadrapole ISOSPIN AVERAGE
#beta[1] = 0.2425  # Deformation parameter, quadrapole
#beta[2] = 0.2425
#beta[3] = 0.235

#Ca40
#E = 17.3
#Z = 20
#A = 40
#
#E_f = -4.71
#v = [0, 61.6, 0.0072, 0.000018, 7e-9]
#w = [0, 14.0, 76.0]
#d = [0, 15.2, 0.0205, 13.4]
#v_so = [0, 6.1, 0.0040]
#w_so = [0, -3.1, 160]
#
#R_c = 1.285 * A ** othird
#R_v = 1.206 * A ** othird
#A_v = 0.582
#R_d = 1.295 * A ** othird
#A_d = 0.535
#R_so = 1.01 * A ** othird
#A_so = 0.60
#
#no_betas = 3
#beta = np.zeros(no_betas)
#beta[0] = 0.361  # Deformation parameter, quadrapole ISOSPIN AVRAGE
#beta[1] = 0.107  # Deformation parameter, hexapole
#beta[2] = 0.191
#


#Fe54
#E=17.9
#Z = 26
#A = 54
#
#E_f = -6.96
#v = [0, 63.0, 0.0072, 0.000018, 7e-9]
#w = [0, 15.2, 78.0]
#d = [0, 15.4, 0.0223, 10.9]
#v_so = [0, 6.1, 0.0040]
#w_so = [0, -3.1, 160]
#
#R_c = 1.285 * A ** othird
#R_v = 1.186 * A ** othird
#A_v = 0.663
#R_d = 1.282 * A ** othird
#A_d = 0.545
#R_so = 1.00 * A ** othird
#A_so = 0.58
#
#no_betas = 4
#beta = np.zeros(no_betas)
#beta[0] = 0.49  # Deformation parameter, quadrapole ISOSPIN AVERAGE
#beta[1] = 0.24  # Deformation parameter, hexapole
#beta[2] = 0.22
#beta[3] = 0.46
#

#Sn120
E = 16.0
glob = False
Z = 50
A = 120
E_f = -8.23
v = [0, 65.5, 0.0077, 0.000019, 7e-9]
w = [0, 16.0, 84.0]
d = [0, 19.3, 0.0206, 13.0]
v_so = [0, 6.2, 0.0040]
w_so = [0, -3.1, 160]

R_c = 1.231 * A ** othird
R_v = 1.225 * A ** othird
A_v = 0.662
R_d = 1.269 * A ** othird
A_d = 0.605
R_so = 1.07 * A ** othird
A_so = 0.60

no_betas = 3
beta = np.zeros(no_betas)
beta[0] = 0.123  # Deformation parameter, quadrapole ISOSPIN AVRAGE
beta[1] = 0.157 # Deformation parameter, hexapole
beta[2] = -0.343

#Coulomb beta: Right now average
beta_C = 0
for i in range(0,no_betas):
    beta_C += beta[i]/no_betas


sum_betas = 0
for i in range (0,no_betas):
    sum_betas += beta[i]
print ("sum_B =", sum_betas)


#ALTERING RADII AND DIFFUSIVE PARAMETERS BASED ON BETAS
N = A-Z
alpha = (N - Z)/A
scale = 1.0              # Scale of modification (% of formula to apply)
delta_r = 1.0
delta_rc = 1 + no_betas*beta_C * scale* np.abs(beta_C)/4.0/np.pi 

for i in range(0, no_betas):  #Changed to +
    delta_r += beta[i] * scale * beta[i] / 4.0 / np.pi


print("del_r=", delta_r)

#MODIFIED PARAMETERS Changed A to multiplication
if glob==False:
   R_v_m = R_v * delta_r
   A_v_m = A_v *delta_r
   R_d_m = R_d * delta_r
   A_d_m = A_d *delta_r
   R_so_m = R_so * delta_r
   A_so_m = A_so * delta_r
   if coul_beta:
       R_c_m = R_c * delta_rc*delta_rc # Term is squared
   else:
       R_c_m = R_c * delta_r*delta_r



r_arr = np.linspace(1.0e-4, 18.0, 6000)   #defining radius, from 0 to 18 fm

#PARAMETERS FOR GLOBAL POTENTIAL
def global_v(N,Z,A):
    a = np.zeros(5)
    a[1]= 59.3 + 21*alpha - 0.024*A
    a[2]= 0.007067 + 4.23e-6 * A
    a[3]= 1.729e-5 + 1.136e-8 * A
    a[4] = 7e-9
    return a
def global_w(A):
    b = np.zeros(3)
    b[1] = 14.667 + 0.009629 * A
    b[2] = 73.55 + 0.0795 * A
    return b
def global_d(N,Z,A):
    c = np.zeros(4)
    c[1] = 16 + 16*alpha
    c[2] = 0.018 + 0.003802/(1 + np.exp((A-156.0)/8.0))
    c[3] = 11.5
    return c
def global_v_so(A):
    temp = np.zeros(3)
    temp[1] = 5.922 + 0.0030 * A
    temp[2] = 0.0040
    return temp
def global_w_so():
    temp = np.zeros(3)
    temp[1] = -3.1
    temp[2] = 160
    return temp
def global_E_f(A):
    temp = -8.4075 + 0.01378*A
    return temp
def global_R_v(A):
    temp = 1.3039 - 0.4054 * A**(-1./3.)
    return temp
def global_A_v(A):
    temp = 0.6778 - 1.487e-4 * A
    return temp
def global_R_d(A):
    temp = 1.3424 - 0.01585 * A**othird
    return temp
def global_A_d(A):
    temp = 0.5187 + 5.205e-4 * A
    return temp
def global_R_so(A):
    temp = 1.1854 - 0.647 * A**(-othird)
    return temp
def global_A_so():
    return 0.59
def global_R_c(A):
    return 1.198 + 0.697 * A**(-2./3.) + 12.994 * A**(-5./3.)
def global_coul_corr(N,Z,A,r_c):
    return 1.73*Z/(r_c*A**othird)

if glob == True:
    r_c = global_R_c(A)
    V_c = global_coul_corr(N,Z,A,r_c)
    v = global_v(N,Z,A)
    w = global_w(A)
    d = global_d(N, Z, A)
    v_so = global_v_so(A)
    w_so = global_w_so()
    E_f = global_E_f(A)
    R_v = global_R_v(A) * A**othird
    A_v = global_A_v(A)
    R_d = global_R_d(A) * A**othird
    A_d = global_A_d(A)
    R_so = global_R_so(A) *A**othird
    A_so = global_A_so()
    R_c = global_R_c(A) * A**othird
    R_v_m = R_v * delta_r
    A_v_m = A_v *delta_r
    R_d_m = R_d * delta_r
    A_d_m = A_d *delta_r
    R_so_m = R_so * delta_r
    A_so_m = A_so * delta_r
    if coul_beta:
        R_c_m = R_c * delta_rc*delta_rc # Term is squared
    else:
        R_c_m = R_c * delta_r*delta_r

parameters = [Z, E, E_f, v, w, d, v_so, w_so, R_c, R_v, A_v, R_d, A_d, R_so, A_so]



#Coulomb constants
e = 1
Eo = 0.0553 # units = e^2 / (Mev x fm)
k = 1 / (4 * np.pi * Eo)



#Spin-orbit Dot Product
def dot_product():
    return 1



#POTENTIAL DEPTHS
# Below are formulas straight from the global paper of Delaroch and Koenig
#Volume depths
def V_v(v, E_f, E):
    volVarReal = v[1] * (1 - v[2]*(E - E_f) + v[3]*(E-E_f)**2 - v[4]*(E-E_f)**3)
    if glob==True:  # Couloumb Correction
        volVarReal=volVarReal+V_c*v[1]*(v[2] -2*v[3]*(E-E_f) + 3*v[4]*(E-E_f)**2)
    return volVarReal

def W_v(w, E_f, E):
    volVarIm = w[1] * (E-E_f)**2 / ((E-E_f)**2 + w[2]**2)
    return volVarIm

#Surface depths
def W_d(d, E_f, E):
    surVarIm = d[1] * (E-E_f)**2 * math.exp(-d[2]*(E-E_f)) / ((E-E_f)**2 + d[3]**2)
    return surVarIm

#Spin-orbit depths
def V_so(v_so, E_f, E):
    soVarReal = v_so[1] * math.exp(-v_so[2] * (E-E_f))
    return soVarReal

def W_so(w_so, E_f, E):
    soVarIm = w_so[1] * (E-E_f)**2 / ((E-E_f)**2 + w_so[2]**2)
    return soVarIm

#FORM FUNCTIONS

# Standard optical potential functions
#Woods-Saxon function
def fws(r, R_v, A_v):
    print("Param:",r,R_v,A_v)
    fwsVol = 1.0 / (1.0 + np.exp((r-R_v)/A_v))
    return fwsVol

#d/dr of Woods-Saxon functions
def fwsDerivSur(r, R_d, A_d): # Mutiplied by -4a (derivative has negative also)
    dfwsdrSur = 4.0*A_d*np.exp((r-R_d)/A_d)/(A_d*(1 + np.exp((r-R_d)/A_d))**2)
    return dfwsdrSur
#Includes 2/r term
def fwsDerivSO(r, R_so, A_so):
    dfwsdrSO = -np.exp((r-R_so)/A_so)/(A_so*(1 + np.exp((r-R_so)/A_so))**2) * 2.0/r
    return dfwsdrSO

#Volume Terms (depth x Woods-Saxon)
volTerm = np.multiply(V_v(v,E_f,E),fws(r_arr,R_v, A_v))
volTermIm = np.multiply(W_v(w,E_f,E), fws(r_arr, R_v, A_v))
volTermIm2 = np.multiply(1j, volTermIm)
VOL = np.add(volTerm, volTermIm2)

#Surface term (depth x Woods-Saxon)
surf= np.multiply(W_d(d,E_f,E),fwsDerivSur(r_arr, R_d, A_d))
SURF = np.multiply(1j, surf)

#Spin-orbit terms (depth x Woods-Saxon)
soTerm = np.multiply(V_so(v_so,E_f,E), fwsDerivSO(r_arr, R_so, A_so))
soTermIm = np.multiply(W_so(w_so,E_f,E), fwsDerivSO(r_arr, R_so, A_so))
soTermIm2 = np.multiply(1j, soTermIm)

SPIN_ORBIT = np.add(soTerm, soTermIm2)

#Coulomb term
def F_c(r_arr, Z, e, R_c):
   if r_arr > R_c:
         Fcoul = k * Z * e**2 / r_arr
   else: Fcoul = k * Z * e**2/ (2.0 * R_c ) * (3.0 - r_arr**2 / R_c**2)
   return Fcoul

vF_c = vectorize(F_c)
COUL = vF_c(r_arr,Z,e,R_c)

#OPTICAL POTENTIAL U(r) (sum of individual terms)
#U Central
def U_c(r_arr, parameters, VOL, SURF, COUL):

    return -1.0*(np.add(np.add(SURF, VOL), COUL))

U_solC = U_c(r_arr, parameters, VOL, SURF, COUL).real

#U spin-orbit
def U_so(r_arr, parameters, SPIN_ORBIT):
    return SPIN_ORBIT

U_solSO = U_so(r_arr, parameters, SPIN_ORBIT).real




#VOLUME INTERGALS

r1 = 0.0
r2 = 100.0

theta1 = 0.0
theta2 = pi


# Here I take volume integrals to normalize results
phi1 = 0.0
phi2 = 2.0 * pi
vp = [r1, r2, theta1, theta2, phi1, phi2]
Vv = V_v(v, E_f, E)
#Volume
def diff_vol1(r, theta, phi):       #equation to be integrated (Woods saxon)
    total = (r**2 * (1.0 + exp((r-R_v)/A_v))**(-1.0) * sin(theta))
#    print("r,theta,phi,total:",r,theta,phi,total)
    return total
def diff_vol1_m(r, theta, phi):     #with modified params
    total = (r**2 * (1.0 + exp((r-R_v_m)/A_v_m))**(-1.0) * sin(theta))
#    print("r,theta,phi,total:",r,theta,phi,total,R_v_m,A_v_m)
    return total

def vol_int_v(vp):
    total = tplquad(diff_vol1,phi1,phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total
def vol_int_v_m(vp):               #volume term integral modified
    total = tplquad(diff_vol1_m, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total

#Surface
def diff_vol2(r, theta, phi):       #equation to be integrated (Woods saxon deriv)
   return -np.exp((r-R_d)/A_d)/(A_d*(1 + np.exp((r-R_d)/A_d))**2) *r**2 * sin(theta)
def diff_vol2_m(r, theta, phi):      #with modified params
   return -np.exp((r-R_d_m)/A_d_m)/(A_d_m*(1 + np.exp((r-R_d_m)/A_d_m))**2) *r**2 * sin(theta)


def vol_int_d(vp):
    total = tplquad(diff_vol2, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total
def vol_int_d_m(vp):               #volume integral modified
    total = tplquad(diff_vol2_m, phi1,phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total


#Spin-orbit
def diff_vol3(r, theta, phi):       #equation to be integrated (Woods saxon deriv) Needs 2/r factor
   return -np.exp((r-R_so)/A_so)/(A_so*(1 + np.exp((r-R_so)/A_so))**2) *r**2 * sin(theta)/r
def diff_vol3_m(r, theta, phi):       #with modified params
   return -np.exp((r-R_so_m)/A_so_m)/(A_so_m*(1 + np.exp((r-R_so_m)/A_so_m))**2) *r**2 * sin(theta)/r


def vol_int_so(vp):
    total = tplquad(diff_vol3, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total
def vol_int_so_m(vp):               #volume term integral modified
    total = tplquad(diff_vol3_m, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y,z: r1, lambda y,z: r2)
    return total


# Coulomb has to be handled special because of long range
#Coulomb
def f_c(rr, r):
    temp = 0.0
    if rr <= r:
        temp = 0.5/r*(3.0 - rr**2/r**2)
    else:
        temp = 1.0/rr
    return temp
vol_int_c = quad(f_c, 0.0, 100.00, args=R_c)
vol_int_c_m = quad(f_c, 0.0, 100.00, args=R_c_m)




#Vol Integral Ratios
vol_ratio_v = vol_int_v(vp)[0] / vol_int_v_m(vp)[0]
vol_ratio_d = vol_int_d(vp)[0] / vol_int_d_m(vp)[0]
vol_ratio_so = vol_int_so(vp)[0] / vol_int_so_m(vp)[0]
vol_ratio_c = vol_int_c[0] / vol_int_c_m[0]

print("vol ratio ",vol_ratio_v, vol_int_v(vp)[0],vol_int_v_m(vp)[0])
# print(vol_ratio_d)
# print(vol_ratio_so)
# print(vol_ratio_c)


print("ECIS Koning-Delaroche:")
print("{:10.5f}{:10.5f}{:10.5f}".format(V_v(v, E_f, E), R_v/A**othird, A_v))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_v(w, E_f, E), R_v/A**othird, A_v))
print("{:10.5f}{:10.5f}{:10.5f}".format(0.0, R_d/A**othird, A_d))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_d(d, E_f, E), R_d/A**othird, A_d))
print("{:10.5f}{:10.5f}{:10.5f}".format(V_so(v_so, E_f, E), R_so/A**othird, A_so))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_so(w_so, E_f, E), R_so/A**othird, A_so))
print("{:10.5f}{:10.5f}{:10.5f}".format(R_c/A**othird, 0.0, 0.0))
print("{:10.5f}{:10.5f}{:10.5f}".format(0.0, 0.0, 0.0))


print("ECIS Distorted Potential:")
print("{:10.5f}{:10.5f}{:10.5f}".format(V_v(v, E_f, E)*(vol_ratio_v/delta_r**3), R_v_m/A**othird, A_v_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_v(w, E_f, E)*(vol_ratio_v/delta_r**3), R_v_m/A**othird, A_v_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(0.1, R_d_m/A**othird, A_d_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_d(d, E_f, E)*vol_ratio_d/delta_r**3, R_d_m/A**othird, A_d_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(V_so(v_so, E_f, E)*(vol_ratio_so/delta_r**3), R_so_m/A**othird, A_so_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(W_so(w_so, E_f, E)*(vol_ratio_so/delta_r**3), R_so_m/A**othird, A_so_m))
print("{:10.5f}{:10.5f}{:10.5f}".format(R_c_m*(vol_ratio_c/delta_rc**3)/A**othird, 0.0, 0.0))
print("{:10.5f}{:10.5f}{:10.5f}".format(0.0, 0.0, 0.0))

