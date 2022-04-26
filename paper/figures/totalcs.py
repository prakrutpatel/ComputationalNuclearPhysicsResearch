import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import product, groupby
import os
from paper.figures.functions import *

ROOT_DIR = os.getcwd()
inputs = ROOT_DIR + "\inputs"
os.chdir(inputs)

with open("totalcs.txt") as f:
    file = f.readlines()  
    elements = file[0].split('\t')
    elements[-1] = elements[-1].strip()
    isotope = file[1].split('\t')
    isotope[-1] = isotope[-1].strip()
    energy = file[2].split('\t')
    energy[-1] = energy[-1].strip()
    pn = file[3].strip()
    order = file[4].split('\t')
    order[-1] = order[-1].strip()
groupby(energy, lambda x: x == '')
energy = [list(group) for k, group in groupby(energy, lambda x: x == '') if not k]

# building datapaths and filenames for importing

def build_path():
    global datapath, elements, isotope, energy, ROOT_DIR
    for i in range(0, len(elements)):
        datapath.append(ROOT_DIR + "\\Ecis_Data\\" + elements[i] + "\\" + isotope[i] + "\\")  
    datapath = [[el] for el in datapath]
    for i in range(0, len(datapath)):
        datapath[i] = list(product(datapath[i], energy[i]))
        datapath[i] = list(map(''.join, datapath[i]))
    return datapath

def build_files():
    global files, elements, isotope, energy, pn, order
    for i in range(0, len(isotope)):
        for j in range(0, len(energy[0])):    
            for k in range(0, len(order)):    
                files.append(elements[i] + isotope[i] + energy[i][j] + pn + '_' + order[k] + '.ecis')
    return files

datapath = []
build_path()
files = []
files = np.array_split(list(chunk(build_files(), len(order))), len(elements))

# importing and converting data into plottable format

totalcs = []
for i in range(0, len(datapath)):  
    for j in range(0, len(datapath[i])):
        os.chdir(datapath[i][j])
        for k in range(0, len(order)):
            import_data(files[i][j][k], '==> total cross section', totalcs)
energyplot = energy
totalcs = np.array_split(totalcs, len(energy))
for i in range(0, len(elements)):
    for j in range(0, len(energy[i])):
        energyplot[i][j] = float(energyplot[i][j])
energyplot = np.array_split(list(np.repeat(energyplot, len(order))), len(energyplot))

# scatter-plot and regression line for totalcs

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
fig = plt.figure('''figsize=(4.0, 5.0)''')
plt.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=True, labelright=False, direction='in')
plt.xlabel(xlabel=r'$E_{Lab}$ (MeV)')
plt.ylabel(ylabel=r'σ (mb)')
plt.tick_params(which='major', length=8, width=2)
plt.tick_params(which='minor', length=4, width=1)
colors = np.array(['blue', 'red', 'green'])
for i in range(0, len(elements)):
    position = list(range(len(order)))
    position1 = list.copy(position)
    for j in range(1, len(energy[i])):
        position.extend(position1)
    plt.scatter(energyplot[i], (250*i)+totalcs[i], color=colors[position])
    plt.plot(energyplot[i], best_fit(energyplot[i], (250*i)+totalcs[i]), color='black', linewidth=0.8, linestyle='--')

plt.show()

