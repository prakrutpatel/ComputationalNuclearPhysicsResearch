import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import os

ROOT_DIR = os.getcwd()
inputs = ROOT_DIR + "\inputs"
os.chdir(inputs)

# input data from template, change the open statement below to the desirable template file

with open("3panelcs.txt") as f:
    file = f.readlines()  
    elements = file[0].strip()
    isotope = file[1].strip()
    energy = file[2].split('\t')
    energy[-1] = energy[-1].strip()
    states = file[3].split('\t')
    states[-1] = states[-1].strip()
    pn = file[4].strip()
    order = file[5].split('\t')
    order[-1] = order[-1].strip()
    num = 4

# functions used for inputing data

def build_path():
    global datapath, elements, isotope, energy, ROOT_DIR
    for i in range(0, len(energy)):
        datapath.append(ROOT_DIR + "\\Ecis_Data\\" + elements + "\\" + isotope + "\\" + energy[i])  
    return datapath

def build_files():
    global files, elements, isotope, energy, pn, order
    for i in range(0, len(energy)):
        for j in range(0, len(order)):
            files.append(elements + isotope + energy[i] + pn + '_' + order[j] + '.ecis')
    return files

def conv(*args):
    conv_dict ={}
    for num in args:
        conv_dict[num]=lambda x:float(x.replace(b'D',b'e'))
    return conv_dict

def find_trigger(filename, word):
    if 'last' in filename:
        loc = []
        with open(filename) as f:
            counter = 0
            page = f.readlines()
            for line in page:
                counter += 1
                if word in line:
                    loc.append(counter)
        loc = loc[num:]
        loc[0] = loc[0] - 2
        loc = [el + 4 for el in loc]
        return loc
    
    else:
        loc = []
        with open(filename) as f:
            counter = 0
            page = f.readlines()
            for line in page:
                counter += 1
                if word in line:
                    loc.append(counter)
        loc[0] = loc[0] - 2
        loc = loc[:num]
        loc = [el + 4 for el in loc]
        return loc
                
def import_data(filename, word, array):
    find_trigger(filename, word)
    array.append(np.loadtxt(filename, skiprows=loc[0], max_rows=169, usecols=(2), converters =conv(1)))
    for i in range(1, len(loc)):    
        array.append(np.loadtxt(filename, skiprows=loc[i], max_rows=169, usecols=(1), converters =conv(1)))
    return array

# importing data from ecis files in specific directories

datapath = []
build_path()
files = []
files = np.array_split(build_files(), len(energy))
angle = np.array(range(8, 177))

cs = []
for i in range(0, len(datapath)):
    os.chdir(datapath[i])
    for j in range(0, len(files[i])):
        loc = find_trigger(files[i][j], 'elastic scattering')
        import_data(files[i][j], 'elastic scattering', cs)
cs = np.array_split(cs, len(energy))
for i in range(0, len(cs)):
    cs[i] = np.array_split(cs[i], len(order))

# cs[i][j][k] = cs[energy][order][state]
# code for plotting below

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(1,3, figsize=(6.8, 4.2), sharey=True)
fig.subplots_adjust(wspace=.04, hspace=.04)
axs = axs.ravel()
axs[0].set_yscale('log')
axs[0].set_yticks([1, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9])
axs[0].yaxis.set_minor_locator(LogLocator(base=10, subs=[1,2,3,4,5,6,7,8,9], numticks=15))
colors = np.array(['blue', 'red', 'green'])
linestyles = np.array(['-', '-.', '-'])
for i in range(0, len(energy)):
    for j in range(0, len(order)):
        for k in range(0, len(states)):  
            axs[k].plot(angle, (10**(-1.5*i))*cs[i][j][k], color=colors[j], linestyle=linestyles[j])
            axs[k].set_xlabel(r'Θ$_{c.m.}$(deg)')
            axs[k].xaxis.set_minor_locator(MultipleLocator(10))
            axs[k].xaxis.set_major_locator(MultipleLocator(50))
            axs[k].tick_params(which='major', length=7, width=1.6)
            axs[k].tick_params(which='minor', length=3, width=.8)
            axs[k].tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=False, labelright=False, direction='in')

axs[0].tick_params(which='both', labelleft=True)
axs[0].set_ylabel(r'dσ/dΩ (mb/sr)')


plt.show()










