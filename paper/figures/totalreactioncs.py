import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import product, groupby
import os
from paper.figures.functions import *

ROOT_DIR = os.getcwd()
inputs = ROOT_DIR + "\inputs"
os.chdir(inputs)

with open("totalreactioncs.txt") as f:
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

reactioncs = []
for i in range(0, len(datapath)):  
    for j in range(0, len(datapath[i])):
        os.chdir(datapath[i][j])
        for k in range(0, len(order)):
            import_data(files[i][j][k], 'total reaction cross section', reactioncs)
energyplot = energy
reactioncs = np.array_split(reactioncs, len(energy))
for i in range(0, len(elements)):
    for j in range(0, len(energy[i])):
        energyplot[i][j] = float(energyplot[i][j])
energyplot = np.array_split(list(np.repeat(energyplot, len(order))), len(energyplot))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
fig = plt.figure()
plt.tick_params(which='major', length=7, width=1.6)
plt.tick_params(which='minor', length=3, width=.8)
plt.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=True, labelright=False, direction='in')
plt.xlabel(xlabel=r'$E_{Lab}$ (Mev)')
plt.ylabel(ylabel=r'σ (mb)')
colors = np.array(['blue', 'red', 'green'])
for i in range(0, len(elements)):
    position = list(range(len(order)))
    position1 = list.copy(position)
    for j in range(1, len(energy[i])):
        position.extend(position1)
    plt.scatter(energyplot[i], (250*i)+reactioncs[i], color=colors[position])
    plt.plot(energyplot[i], best_fit(energyplot[i], (250*i)+reactioncs[i]), color='black', linewidth=0.8, linestyle='--')

plt.show()

