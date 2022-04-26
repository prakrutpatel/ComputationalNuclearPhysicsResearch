# This program will try to recalculate a DWBA potential from  astandard elastic optical potential and recalculate it to work
# in a coupled channel code. It will read in the plot.param file. Inputs also include the distortion parameters
# Mix of Charlie, Steve and PK
import re
import sys
import numpy as np
from decimal import *
from numpy import vectorize, angle
from scipy.integrate import tplquad, quad
from mendeleev import element
import beta_changer as bp
Initial_cond=2.0 # initial potential surface (real) for Koning, Delaroche
# BOOLEANS FOR BETA CHANGER
bool_beta_changer = True
new_beta_changer= True
# The old beta changer (first one) made only one beta (the first) for each distortion (2+,3-,4+)
# It also changed the ECIS file to recognize this. Every 2+ level in C12, for example, had the same
# beta value. This was duplicated in ECIS. For calculating r_delta the other numbers
# were removed

# The new beta changer keeps all the beta values found. Keeps templete the same, Keeps ECIS the same.
# It is simplier, it only changes how r_delta is calculated. It takes a weighted average value instead
# For example it has a weighted average for both the 2+ levels

#For New Beta Changer to Work Both Have to be true
#saving arguments from file and argument line
data_list = sys.argv[1]
E = float(sys.argv[2])
template = sys.argv[3]
case = sys.argv[4]
new_data_list = data_list.replace('.txt','')
beta_sum = data_list.replace('.txt','.betas')
print("Status ECIS:",case)
Z = 0
A = 0
no_betas = 0
v =  np.zeros(5)
w = np.zeros(5)
d = np.zeros(5)
v_so = np.zeros(5)
w_so = np.zeros(5)
b = np.zeros(5)
E_f = 0.0
R_c = 0.0
R_v = 0.0
A_v = 0.0
R_d = 0.0
A_d = 0.0
R_so = 0.0
A_so = 0.0
energy_trig_wepp=0
b_str=[]
# INPUTS:
# Is the Coulumb field distorted
coul_beta = False
# Do I use a global potential from Koenig,   Delaroche
othird = 1. / 3.

#to split a string in a specific format
def my_split(s):
    return (re.split(r'(\d+)', s))

#saving incoming template file into an array of strings
with open(template) as f:
    data = f.read().splitlines()
with open(data_list) as r:
    tempo=r.read().splitlines()
#saving spin and ground state from input file
spin=tempo[5]
grd=tempo[2]
grd_split=grd.split()

#********************************************
#Saving betas from template file, different spins have different format therefore I created two case for each format
no_betas_line=data[0].split()
no_betas=int(no_betas_line[0])-1
beta = np.zeros(no_betas)
BOOL_HOLDER=[]
for u in range(0,no_betas):
    BOOL_HOLDER.append("True")
if spin == 'Vibrational':
    beta_trig=(no_betas*2)+3
    for m in range(0,(no_betas*2)-1):
        if m%2 == 0:
            beta_split=data[beta_trig+m].split()
            b_str.append(float(beta_split[2]))
    x_b = np.array(b_str)
    b = x_b.astype(np.float)
    for i in range(0, no_betas):
        beta[i] = b[i]
offset = 0    #only for rotational
if spin == 'Rotational':
    num_ones=0
    for g in range(2,2+no_betas+num_ones):
        temp_str = data[g][5:9]
        if ((temp_str != ' 00 ') and (temp_str != '0 0 ') and temp_str != ('0  0') and temp_str != (' 0 0')):
            offset = offset + 3
            if g != 1+no_betas:
                 num_ones=num_ones+3
    beta_line=data[no_betas+3+offset]
    beta_list=beta_line.split()
    for i in range(0,len(beta_list)):
        b_str.append(beta_list[i])
    x_b = np.array(b_str)
    b = x_b.astype(np.float)
    for i in range(0, no_betas):
        beta[i] = b[i]
#******************************
if bool_beta_changer:
    bp.beta_changer(template, spin, offset, new_beta_changer, case, BOOL_HOLDER, data, E, beta)
    if bp.beta_changer.new_beta == 1:
        beta=bp.beta_changer.new_beta_array.copy()
        no_betas=len(bp.beta_changer.new_beta_array)
        if case=='on':
            print("Beta changer: ", no_betas, '  ', bp.beta_changer.new_beta_array)
            if new_beta_changer:
               print("A weighted average of the Betas has been taken")
    else:
        if case=='on':
            print("No Betas changed in Betachanger: ", no_betas, '  ', beta)
else:
    if case=='on':
        print("No_betas ", no_betas, '   ', beta)
#*************************************************************************8

with open(data_list) as f:#saving incoming data file into an array of strings
    lines = f.read().splitlines()
    Element = lines[0]
    glob = lines[1]
    ground_state=lines[2]
    #there can be 2 values for glob, each saving different values
    if glob == 'True':
        Z = int(lines[3])
        A = int(lines[4])
    if glob == 'False':
        Z = int(lines[3])
        A = int(lines[4])
        E_f = float(lines[6])
        R_c = float(lines[12]) * A ** othird
        R_v = float(lines[13]) * A ** othird
        A_v = float(lines[14])
        R_d = float(lines[15]) * A ** othird
        A_d = float(lines[16])
        R_so = float(lines[17]) * A ** othird
        A_so = float(lines[18])
        v_str=lines[7]
        v_str = v_str.split(',')
        x_v = np.array(v_str)
        v = x_v.astype(np.float)
        w_str=lines[8]
        w_str = w_str.split(',')
        x_w = np.array(w_str)
        w = x_w.astype(np.float)
        d_str=lines[9]
        d_str = d_str.split(',')
        x_d = np.array(d_str)
        d = x_d.astype(np.float)
        v_so_str=lines[10]
        v_so_str = v_so_str.split(',')
        x_v_so = np.array(v_so_str)
        v_so = x_v_so.astype(np.float)
        w_so_str=lines[11]
        w_so_str = w_so_str.split(',')
        x_w_so = np.array(w_so_str)
        w_so = x_w_so.astype(np.float)
    if glob == "Weppner":
        Z = int(lines[3])
        A = int(lines[4])
        potential_trig=str(E)+' MeV'
        for a in range(0,len(lines)):
            if potential_trig in lines[a]:
                energy_trig_wepp = a
        R_c = float(lines[energy_trig_wepp+7].split()[0]) * A ** othird
        R_v = float(lines[energy_trig_wepp+1].split()[1]) * A ** othird
        A_v = float(lines[energy_trig_wepp+1].split()[2])
        R_d = float(lines[energy_trig_wepp+3].split()[1]) * A ** othird
        A_d = float(lines[energy_trig_wepp+3].split()[2])
        R_so =float(lines[energy_trig_wepp+5].split()[1]) * A ** othird
        A_so =float(lines[energy_trig_wepp+5].split()[2])
                
#assign cond values based on spin and case
if spin == 'Vibrational':
    if case == 'on':
        cond_1 = "FFFFTFFFFFTTTTFFFFFFFFFFFFFFFTFFFFFFFFFFFFFFFFFFFF"
        cond_2 = "FFFFFFFFFFFFFFFFTTTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    if case == 'mid' or case == 'off':
        cond_1 = "FFFFTFFFFFTTTTFFFFFFTFFFFFFFFFTTFFFFFFFFFFFFFFFFFF"
        cond_2 = "FFFFFFFFFFFFFFFFTTTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        angles = np.load(new_data_list+'angles.npy')
        cs = np.load(new_data_list+'cs.npy')
        polar = np.load(new_data_list+'polar.npy')
        rcs = np.load(new_data_list+'rcs.npy')
if spin == 'Rotational':
    if case == 'on':
        cond_1 = "TFFFTFFFFFTTTTFFFFFFFFFFFFFFFTFFFFFFFFFFFFFFFFFFFF"
        cond_2 = "FFFFFFFFFFFFFFFFTTTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    if case == 'mid' or case == 'off':
        cond_1 = "TFFFTFFFFFTTTTFFFFFFTFFFFFFFFFTTFFFFFFFFFFFFFFFFFF"
        cond_2 = "FFFFFFFFFFFFFFFFTTTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        angles = np.load(new_data_list+'angles.npy')
        cs = np.load(new_data_list+'cs.npy')
        polar = np.load(new_data_list+'polar.npy')
        rcs = np.load(new_data_list+'rcs.npy')
#Using chemistry package to import chemical values for the element being used
arr = (my_split(Element))
p_name = element(arr[0])
mass_num = int(arr[1])
for iso in p_name.isotopes:
    if iso.mass_number == mass_num:
        element_mass=iso.mass
        atomic_number = float(iso.atomic_number)
#Coulomb beta: Right now average
beta_C = 0
for i in range(0, no_betas):
    beta_C += beta[i] / no_betas

sum_betas = 0
for i in range (0, no_betas):
    sum_betas += beta[i]

# ALTERING RADII AND DIFFUSIVE PARAMETERS BASED ON BETAS
N = A - Z
alpha = (N - Z) / A
scale = 1.0  # Scale of modification (% of formula to apply)
delta_r = 1.0
delta_rc = 1 - no_betas * beta_C * scale * beta_C / 4.0 / np.pi

for i in range(0, no_betas):  # Changed to +
    delta_r -= beta[i] * scale * beta[i] / 4.0 / np.pi
# print sum of square beta file
if case == 'off':
    with open(beta_sum,"w") as f_beta:
        f_beta.write(str(1-delta_r)+'\n')
    f_beta.close()
    
r_arr = np.linspace(1.0e-4, 18.0, 6000)  # defining radius, from 0 to 18 fm

#Use hard_data to create fit data. 2x2 array is used. First element is parameter, second is line in potential block and third is the non-linear regression fit
#Use site 'http://www.xuru.org/rt/NLR.asp#CopyPaste' to create non-linear regression fit

hard_data = [[9,3,'beta_fit*(34.5775+2.66384*x+0.0179569*x**2-0.000082700*x**3)+V_d_0'],
            [13,4,'beta_fit**0.5*(-8.69164-0.0367497*x-0.00275055*x**2+0.000013281*x**3)+W_d_0']]


# PARAMETERS FOR GLOBAL POTENTIAL
def global_v(N, Z, A):
    a = np.zeros(5)
    a[1] = 59.3 + 21 * alpha - 0.024 * A
    a[2] = 0.007067 + 4.23e-6 * A
    a[3] = 1.729e-5 + 1.136e-8 * A
    a[4] = 7e-9
    return a


def global_w(A):
    b = np.zeros(3)
    b[1] = 14.667 + 0.009629 * A
    b[2] = 73.55 + 0.0795 * A
    return b


def global_d(N, Z, A):
    c = np.zeros(4)
    c[1] = 16 + 16 * alpha
    c[2] = 0.018 + 0.003802 / (1 + np.exp((A - 156.0) / 8.0))
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
    temp = -8.4075 + 0.01378 * A
    return temp


def global_R_v(A):
    temp = 1.3039 - 0.4054 * A ** (-1. / 3.)
    return temp


def global_A_v(A):
    temp = 0.6778 - 1.487e-4 * A
    return temp


def global_R_d(A):
    temp = 1.3424 - 0.01585 * A ** othird
    return temp


def global_A_d(A):
    temp = 0.5187 + 5.205e-4 * A
    return temp


def global_R_so(A):
    temp = 1.1854 - 0.647 * A ** (-othird)
    return temp


def global_A_so():
    return 0.59


def global_R_c(A):
    return 1.198 + 0.697 * A ** (-2. / 3.) + 12.994 * A ** (-5. / 3.)


def global_coul_corr(N, Z, A, r_c):
    return 1.73 * Z / (r_c * A ** othird)

if glob == 'True':
    r_c = global_R_c(A)
    V_c = global_coul_corr(N, Z, A, r_c)
    v = global_v(N, Z, A)
    w = global_w(A)
    d = global_d(N, Z, A)
    v_so = global_v_so(A)
    w_so = global_w_so()
    E_f = global_E_f(A)
    R_v = global_R_v(A) * A ** othird
    A_v = global_A_v(A)
    R_d = global_R_d(A) * A ** othird
    A_d = global_A_d(A)
    R_so = global_R_so(A) * A ** othird
    A_so = global_A_so()
    R_c = global_R_c(A) * A ** othird

parameters = [Z, E, E_f, v, w, d, v_so, w_so, R_c, R_v, A_v, R_d, A_d, R_so, A_so]

R_v_m = R_v * delta_r
A_v_m = A_v #* delta_r
R_d_m = R_d * delta_r
A_d_m = A_d #* delta_r
R_so_m = R_so * delta_r
A_so_m = A_so #* delta_r
if coul_beta:
    R_c_m = R_c #* delta_rc * delta_rc  # Term is squared BUT TURNED FEATURE OFF
else:
    R_c_m = R_c #* delta_r * delta_r
# Coulomb constants
e = 1
Eo = 0.0553  # units = e^2 / (Mev x fm)
k = 1 / (4 * np.pi * Eo)


# Spin-orbit Dot Product
def dot_product():
    return 1

# POTENTIAL DEPTHS
# Below are formulas straight from the global paper of Delaroch and Koenig
# Volume depths

def V_v(v, E_f, E):
    volVarReal = v[1] * (1 - v[2] * (E - E_f) + v[3] * (E - E_f) ** 2 - v[4] * (E - E_f) ** 3)
    if glob == 'True':
        volVarReal = volVarReal + V_c * v[1] * (v[2] - 2 * v[3] * (E - E_f) + 3 * v[4] * (E - E_f) ** 2)
    if glob =='Weppner':
        volVarReal = float(lines[energy_trig_wepp+1].split()[0])
    return volVarReal


def W_v(w, E_f, E):
    volVarIm = w[1] * (E - E_f) ** 2 / ((E - E_f) ** 2 + w[2] ** 2)
    if glob =='Weppner':
        volVarIm = float(lines[energy_trig_wepp+2].split()[0])
    return volVarIm

# Surface depths later
# Spin-orbit depths
def V_so(v_so, E_f, E):
    soVarReal = v_so[1] * np.exp(-v_so[2] * (E - E_f))
    if glob =='Weppner':
        soVarReal = float(lines[energy_trig_wepp+5].split()[0])
    return soVarReal

def W_so(w_so, E_f, E):
    soVarIm = w_so[1] * (E - E_f) ** 2 / ((E - E_f) ** 2 + w_so[2] ** 2)
    if glob =='Weppner':
        soVarIm = float(lines[energy_trig_wepp+6].split()[0])
    return soVarIm

#Surface depths involve fits
def V_d(d,e_f,E,fit=False):
    if fit == False:
        if glob!= 'Weppner':
            if case != 'off':
                return 0.0
            else:
                return  Initial_cond
        else:
            return float(lines[energy_trig_wepp+3].split()[0])
    else:#Use hard_data elements to create a custom value to fit our non-linear/linear regression fit
        formula=hard_data[0][2]
        x=mass_num
        beta_fit = 1-delta_r
        V_d_0 = 0.0
        if glob == 'Weppner':
            V_d_0 = float(lines[energy_trig_wepp+3].split()[0])

        f = eval(formula)
        return f

def W_d(d, E_f, E,fit=False):
    #False is default unless other specified specified during the call
    global case
    if fit == False or case =='mid': #default no changes
        if glob != 'Weppner':
            surVarIm = d[1] * (E - E_f) ** 2 * np.exp(-d[2] * (E - E_f)) / ((E - E_f) ** 2 + d[3] ** 2)
        else:
            surVarIm = float(lines[energy_trig_wepp+4].split()[0])
        return surVarIm
    else:#Use hard_data elements to create a custom value to fit our non-linear/linear regression fit
        formula=hard_data[1][2]
        x=mass_num
        beta_fit = 1-delta_r
        W_d_0 = 0.0
        if glob !='Weppner':
            W_d_0= d[1] * (E - E_f) ** 2 * np.exp(-d[2] * (E - E_f)) / ((E - E_f) ** 2 + d[3] ** 2)
        else:
            W_d_0 = float(lines[energy_trig_wepp+4].split()[0])
        k = eval(formula)
        return  k

# FORM FUNCTIONS
# Standard optical potential functions
# Woods-Saxon function
def fws(r, R_v, A_v):
    fwsVol = 1.0 / (1.0 + np.exp((r - R_v) / A_v))
    return fwsVol

# d/dr of Woods-Saxon functions
def fwsDerivSur(r, R_d, A_d):  # Mutiplied by -4a (derivative has negative also)
    dfwsdrSur = 4.0 * A_d * np.exp((r - R_d) / A_d) / (A_d * (1 + np.exp((r - R_d) / A_d)) ** 2)
    return dfwsdrSur

# Includes 2/r term
def fwsDerivSO(r, R_so, A_so):
    dfwsdrSO = -np.exp((r - R_so) / A_so) / (A_so * (1 + np.exp((r - R_so) / A_so)) ** 2) * 2.0 / r
    return dfwsdrSO

# Volume Terms (depth x Woods-Saxon)
volTerm = np.multiply(V_v(v, E_f, E), fws(r_arr, R_v, A_v))
volTermIm = np.multiply(W_v(w, E_f, E), fws(r_arr, R_v, A_v))
volTermIm2 = np.multiply(1j, volTermIm)
VOL = np.add(volTerm, volTermIm2)
# Surface term (depth x Woods-Saxon)
surf = np.multiply(W_d(d, E_f, E), fwsDerivSur(r_arr, R_d, A_d))
SURF = np.multiply(1j, surf)

# Spin-orbit terms (depth x Woods-Saxon)
soTerm = np.multiply(V_so(v_so, E_f, E), fwsDerivSO(r_arr, R_so, A_so))
soTermIm = np.multiply(W_so(w_so, E_f, E), fwsDerivSO(r_arr, R_so, A_so))
soTermIm2 = np.multiply(1j, soTermIm)

SPIN_ORBIT = np.add(soTerm, soTermIm2)
# Coulomb term
def F_c(r_arr, Z, e, R_c):
   if r_arr > R_c:
         Fcoul = k * Z * e ** 2 / r_arr
   else: Fcoul = k * Z * e ** 2 / (2.0 * R_c) * (3.0 - r_arr ** 2 / R_c ** 2)
   return Fcoul

vF_c = vectorize(F_c)
COUL = vF_c(r_arr, Z, e, R_c)

# OPTICAL POTENTIAL U(r) (sum of individual terms)
# U Central
def U_c(r_arr, parameters, VOL, SURF, COUL):

    return -1.0 * (np.add(np.add(SURF, VOL), COUL))


U_solC = U_c(r_arr, parameters, VOL, SURF, COUL).real

# U spin-orbit
def U_so(r_arr, parameters, SPIN_ORBIT):
    return SPIN_ORBIT

U_solSO = U_so(r_arr, parameters, SPIN_ORBIT).real

# VOLUME INTERGALS

r1 = 0.0
r2 = 100.0

theta1 = 0.0
theta2 = np.pi

# Here I take volume integrals to normalize results
phi1 = 0.0
phi2 = 2.0 * np.pi
vp = [r1, r2, theta1, theta2, phi1, phi2]
Vv = V_v(v, E_f, E)

# Volume
def diff_vol1(r, theta, phi):  # equation to be integrated (Woods saxon)
    total = (r ** 2 * (1.0 + np.exp((r - R_v) / A_v)) ** (-1.0) * np.sin(theta))
    return total


def diff_vol1_m(r, theta, phi):  # with modified params
    total = (r ** 2 * (1.0 + np.exp((r - R_v_m) / A_v_m)) ** (-1.0) * np.sin(theta))
    return total


def vol_int_v(vp):
    total = tplquad(diff_vol1, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total


def vol_int_v_m(vp):  # volume term integral modified
    total = tplquad(diff_vol1_m, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total


# Surface
def diff_vol2(r, theta, phi):  # equation to be integrated (Woods saxon deriv)
   return -np.exp((r - R_d) / A_d) / (A_d * (1 + np.exp((r - R_d) / A_d)) ** 2) * r ** 2 * np.sin(theta)


def diff_vol2_m(r, theta, phi):  # with modified params
   return -np.exp((r - R_d_m) / A_d_m) / (A_d_m * (1 + np.exp((r - R_d_m) / A_d_m)) ** 2) * r ** 2 * np.sin(theta)


def vol_int_d(vp):
    total = tplquad(diff_vol2, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total


def vol_int_d_m(vp):  # volume integral modified
    total = tplquad(diff_vol2_m, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total

# Spin-orbit
def diff_vol3(r, theta, phi):  # equation to be integrated (Woods saxon deriv) Needs 2/r factor
   return -np.exp((r - R_so) / A_so) / (A_so * (1 + np.exp((r - R_so) / A_so)) ** 2) * r ** 2 * np.sin(theta) / r

def diff_vol3_m(r, theta, phi):  # with modified params
   return -np.exp((r - R_so_m) / A_so_m) / (A_so_m * (1 + np.exp((r - R_so_m) / A_so_m)) ** 2) * r ** 2 * np.sin(theta) / r

def vol_int_so(vp):
    total = tplquad(diff_vol3, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total


def vol_int_so_m(vp):  # volume term integral modified
    total = tplquad(diff_vol3_m, phi1, phi2, lambda y: theta1, lambda y: theta2,
                            lambda y, z: r1, lambda y, z: r2)
    return total


# Coulomb has to be handled special because of long range
# Coulomb
def f_c(rr, r):
    temp = 0.0
    if rr <= r:
        temp = 0.5 / r * (3.0 - rr ** 2 / r ** 2)
    else:
        temp = 1.0 / rr
    return temp

vol_int_c = quad(f_c, 0.0, 100.00, args=R_c)
vol_int_c_m = quad(f_c, 0.0, 100.00, args=R_c_m)
# Vol Integral Ratios
vol_ratio_v = vol_int_v(vp)[0] / vol_int_v_m(vp)[0]
vol_ratio_d = vol_int_d(vp)[0] / vol_int_d_m(vp)[0]
vol_ratio_so = vol_int_so(vp)[0] / vol_int_so_m(vp)[0]
vol_ratio_c = vol_int_c[0] / vol_int_c_m[0]

#Warning: Only make changes to each parameter from switch_case only, do not change them from code.
#True to make a switch with custom fit, False to use default value from Koning Delaroche potential----- switch_case=[["V_d",True/False],["W_d",True/False],["V_so",True/False],["W_so",True/False]]
switch_case=[["V_d",True],["W_d",True],["V_so",False],["W_so",False]]

print(Element,E,"MeV",no_betas,"states", file=open(new_data_list+".inp", "w"))
print(cond_1, file=open(new_data_list+".inp", "a"))
print(cond_2, file=open(new_data_list+".inp", "a"))
print(data[0], file=open(new_data_list+".inp", "a"))
print(data[1], file=open(new_data_list+".inp", "a"))
print('{:<12}{:<10.4f}{:<10.5f}{:7.5f}{:>10.5f}{:>11.5f}'.format(ground_state,E,0.5,1.00742,element_mass,atomic_number), file=open(new_data_list+".inp", "a"))
for i in range(2, len(data)-2):
    print(data[i], file=open(new_data_list+".inp", "a"))

if case == 'on' or case == 'mid': 
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_v(v, E_f, E)), float(R_v/A**othird), float(A_v)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_v(w, E_f, E)), float(R_v/A**othird), float(A_v)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_d(d,E_f,E)), float(R_d/A**othird), float(A_d)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_d(d, E_f, E)), float(R_d/A**othird), float(A_d)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_so(v_so, E_f, E)), float(R_so/A**othird), float(A_so)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_so(w_so, E_f, E)), float(R_so/A**othird), float(A_so)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(R_c/A**othird), float(0.0), float(0.0)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(0.0), float(0.0), float(0.0)), file=open(new_data_list+".inp", "a"))
    print(data[len(data)-2], file=open(new_data_list+".inp", "a"))

if case == 'off':
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_v(v, E_f, E) * (vol_ratio_v * delta_r ** 3)), float(R_v_m / A ** othird), float(A_v_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_v(w, E_f, E) * (vol_ratio_v * delta_r ** 3)), float(R_v_m / A ** othird), float(A_v_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_d(d, E_f, E, switch_case[0][1])), float(R_d_m / A ** othird), float(A_d_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_d(d, E_f, E, switch_case[1][1])), float(R_d_m / A ** othird), float(A_d_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(V_so(v_so, E_f, E) * (vol_ratio_so * delta_r ** 3)), float(R_so_m / A ** othird), float(A_so_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(W_so(w_so, E_f, E) * (vol_ratio_so * delta_r ** 3)), float(R_so_m / A ** othird), float(A_so_m)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(R_c_m * (vol_ratio_c * delta_rc ** 0) / A ** othird), float(0.0), float(0.0)), file=open(new_data_list+".inp", "a"))
    print("{:10.5f}{:10.5f}{:10.5f}".format(float(0.0), float(0.0), float(0.0)), file=open(new_data_list+".inp", "a"))
    print(data[len(data)-2], file=open(new_data_list+".inp", "a"))
    print("{:5d}{:5d}{:5d}{:5d}{:11.1f}".format(3, 2, 10, 1, 50.0), file=open(new_data_list + ".inp", "a"))
    # Reaction CS Line
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1f}".format('F', 0, 1, 1, 19, 25.0), file=open(new_data_list + ".inp", "a"))
    print("{:8.2f}  {:8.2f}  {:8.2f}".format(0.0, rcs, 10.0), file=open(new_data_list + ".inp", "a"))
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1f}".format('T', 0, len(angles), 1, 0, 0.80),
          file=open(new_data_list + ".inp", "a"))
    for i in range(0, len(angles)):
        print("{:9.1f}{:11.4E}{:5.2f}".format(angles[i], cs[i], (4.00 + ((i / (len(angles) - 1)) * 12))),
              file=open(new_data_list + ".inp", "a"))
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1E}{:11.1f}".format('F', 0, len(angles), 1, 2, 3.0E-1, 1),
          file=open(new_data_list + ".inp", "a"))
    for i in range(0, len(angles)):
        print("{:9.1f}{:11.7f}{:5.2f}".format(angles[i], polar[i], (0.04 + ((i / (len(angles) - 1)) * 0.10))),
              file=open(new_data_list + ".inp", "a"))
    print("     5.E-3     5.E-3", file=open(new_data_list + ".inp", "a"))
    print("    9   13", file=open(new_data_list + ".inp", "a"))
if case == 'mid':
    print("{:5d}{:5d}{:5d}{:5d}{:11.1f}".format(3, 2, 10, 1, 50.0), file=open(new_data_list + ".inp", "a"))
    # Reaction CS Line
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1f}".format('F', 0, 1, 1, 19, 25.0), file=open(new_data_list + ".inp", "a"))
    print("{:8.2f}  {:8.2f}  {:8.2f}".format(0.0, rcs, 10.0), file=open(new_data_list + ".inp", "a"))
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1f}".format('T', 0, len(angles), 1, 0, 0.80),
          file=open(new_data_list + ".inp", "a"))
    for i in range(0, len(angles)):
        print("{:9.1f}{:11.4E}{:5.2f}".format(angles[i], cs[i], (4.00 + ((i / (len(angles) - 1)) * 12))),
              file=open(new_data_list + ".inp", "a"))
    print("{:.1}{:1d}{:3d}{:5d}{:5d}{:15.1E}{:11.1f}".format('F', 0, len(angles), 1, 2, 3.0E-1, 1),
          file=open(new_data_list + ".inp", "a"))
    for i in range(0, len(angles)):
        print("{:9.1f}{:11.7f}{:5.2f}".format(angles[i], polar[i], (0.04 + ((i / (len(angles) - 1)) * 0.10))),
              file=open(new_data_list + ".inp", "a"))
    print("     5.E-3     5.E-3", file=open(new_data_list + ".inp", "a"))
    print("    9   13", file=open(new_data_list + ".inp", "a"))
print(data[len(data)-1], file=open(new_data_list+".inp", "a"))

for i in range(len(switch_case)):
    if (switch_case[i][1]):
        func = switch_case[i][0]+"(d,E_f,E)"
        print(switch_case[i][0],'   ',eval(func),file = open(new_data_list+"param_fit", "a"))

