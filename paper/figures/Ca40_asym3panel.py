import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os, sys


ROOT_DIR = os.getcwd()
inputs = ROOT_DIR + "\inputs"
os.chdir(inputs)

# input data from template, change the open statement below to the desirable template file

levels = []
states = []
with open("3panel_Ca.txt") as f:
    file = f.readlines()  
    levels = file[1].split('\t')
    levels[-1] = levels[-1].strip()
    states = file[2].split('\t')
    states[-1] = states[-1].strip()
element = file[0].strip()
pn = file[3].strip()

# change datapath to the directory where the data files are located.

datapath = ROOT_DIR + "\\" + element + pn + "_Ecis_data"
os.chdir(datapath)

# finding the line number of each state within first files

loc_first = []
with open(element+levels[0]+pn+"_first.ecis") as f:
    counter_first = 0
    page_first = f.readlines()
    for line in page_first:
        counter_first += 1
        if 'elastic scattering' in line:
            loc_first.append(counter_first + 4)
        else:
            continue

num = int(len(loc_first))
loc_first[0] = loc_first[0] - 2

# converts fortran scientific notation d to e

def conv(*args):
    conv_dict ={}
    for num in args:
        conv_dict[num]=lambda x:float(x.replace(b'D',b'e'))
    return conv_dict

# creating the arrays, appending the data, and incorporating trigger word/line counter with loadtxt, state 0 uses c.s/ruther

asym_first_0 = []
asym_first_1 = []
asym_first_2 = []
asym_first_3 = []
asym_middle_0 = []
asym_middle_1 = []
asym_middle_2 = []
asym_middle_3 = []
asym_last_0 = []
asym_last_1 = []
asym_last_2 = []
asym_last_3 = []

for filename in os.listdir(datapath):
    if filename.endswith(".ecis"):
        if('first' in filename):
            angle = np.loadtxt(filename, skiprows=loc_first[0], max_rows=169, usecols=(0), unpack=True)
            asym_first_0.append(np.loadtxt(filename, skiprows=loc_first[0], max_rows=169, usecols=(3), converters =conv(1)))
            asym_first_1.append(np.loadtxt(filename, skiprows=loc_first[1], max_rows=169, usecols=(2), converters =conv(1)))
            asym_first_2.append(np.loadtxt(filename, skiprows=loc_first[2], max_rows=169, usecols=(2), converters =conv(1)))
            asym_first_3.append(np.loadtxt(filename, skiprows=loc_first[3], max_rows=169, usecols=(2), converters =conv(1)))
        elif('middle' in filename):
            loc_middle = []
            with open(filename) as f:
                counter_middle = 0
                page_middle = f.readlines()
                for line in page_middle:
                    counter_middle += 1
                    if 'elastic scattering' in line:
                        loc_middle.append(counter_middle + 4)
                    else:
                        continue
            loc_middle = loc_middle[:num]
            loc_middle[0] = loc_middle[0] - 2
            asym_middle_0.append(np.loadtxt(filename, skiprows=loc_middle[0], max_rows=169, usecols=(3), converters =conv(1)))
            asym_middle_1.append(np.loadtxt(filename, skiprows=loc_middle[1], max_rows=169, usecols=(2), converters =conv(1)))
            asym_middle_2.append(np.loadtxt(filename, skiprows=loc_middle[2], max_rows=169, usecols=(2), converters =conv(1)))
            asym_middle_3.append(np.loadtxt(filename, skiprows=loc_middle[3], max_rows=169, usecols=(2), converters =conv(1)))
        else:
            loc_last = []
            with open(filename) as f:
                counter_last = 0
                page_last = f.readlines()
                for line in page_last:
                    counter_last += 1
                    if 'elastic scattering' in line:
                        loc_last.append(counter_last + 4)
                        loc_last = loc_last[0-num:]
                        loc_last[0] = loc_last[0] - 2
                    else:
                        continue
            asym_last_0.append(np.loadtxt(filename, skiprows=loc_last[0], max_rows=169, usecols=(3), converters =conv(1)))
            asym_last_1.append(np.loadtxt(filename, skiprows=loc_last[1], max_rows=169, usecols=(2), converters =conv(1)))
            asym_last_2.append(np.loadtxt(filename, skiprows=loc_last[2], max_rows=169, usecols=(2), converters =conv(1)))
            asym_last_3.append(np.loadtxt(filename, skiprows=loc_last[3], max_rows=169, usecols=(2), converters =conv(1)))
    else:
        continue

# stacking the lists of arrays into 3D arrays
# for the arrays, the first [i] is for the energy of the incoming particle, and the second is for energy level of the nucleus
# i.e     asym_first[1][0] is the 14.6 MeV proton, elastic (state 0, 0.0+) case

asym_first = np.stack((asym_first_0,asym_first_1,asym_first_2,asym_first_3), axis=1)
asym_middle = np.stack((asym_middle_0,asym_middle_1,asym_middle_2,asym_middle_3), axis=1)
asym_last = np.stack((asym_last_0,asym_last_1,asym_last_2,asym_last_3), axis=1)

# code for plotting below

fig = plt.figure(figsize=(7.4, 4.2))
#plt.rc('font', family='serif', serif='cm10')
plot1 = fig.add_subplot(131)
plot2 = fig.add_subplot(132, sharey=plot1)
plot3 = fig.add_subplot(133, sharey=plot1)
plt.subplots_adjust(wspace=.04, hspace=.04)


plot1.plot(angle, (8)+asym_first[0][0], color='blue')
plot1.plot(angle, (6)+asym_first[1][0], color='blue')
plot1.plot(angle, (4)+asym_first[2][0], color='blue')
plot1.plot(angle, (2)+asym_first[3][0], color='blue')
plot1.plot(angle, (0)+asym_first[4][0], color='blue')
plot1.plot(angle, (8)+asym_middle[0][0], color='red', linestyle='-.')
plot1.plot(angle, (6)+asym_middle[1][0], color='red', linestyle='-.')
plot1.plot(angle, (4)+asym_middle[2][0], color='red', linestyle='-.')
plot1.plot(angle, (2)+asym_middle[3][0], color='red', linestyle='-.')
plot1.plot(angle, (0)+asym_middle[4][0], color='red', linestyle='-.')
plot1.plot(angle, (8)+asym_last[0][0], color='green')
plot1.plot(angle, (6)+asym_last[1][0], color='green')
plot1.plot(angle, (4)+asym_last[2][0], color='green')
plot1.plot(angle, (2)+asym_last[3][0], color='green')
plot1.plot(angle, (0)+asym_last[4][0], color='green')
plot1.set_title('') # change title here
plot1.set_xlabel(r'Θ$_{c.m.}$(deg)')
plot1.set_ylabel(r'A$_{y}$(Θ)')
plot1.xaxis.set_minor_locator(MultipleLocator(10))
plot1.tick_params(which='major', length=8, width=2)
plot1.tick_params(which='minor', length=4, width=1)
plot1.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, 
                  labeltop=False, labelleft=True, labelright=False, direction='in')

plot2.plot(angle, (8)+asym_first[0][1], color='blue')
plot2.plot(angle, (6)+asym_first[1][1], color='blue')
plot2.plot(angle, (4)+asym_first[2][1], color='blue')
plot2.plot(angle, (2)+asym_first[3][1], color='blue')
plot2.plot(angle, (0)+asym_first[4][1], color='blue')
plot2.plot(angle, (8)+asym_middle[0][1], color='red', linestyle='-.')
plot2.plot(angle, (6)+asym_middle[1][1], color='red', linestyle='-.')
plot2.plot(angle, (4)+asym_middle[2][1], color='red', linestyle='-.')
plot2.plot(angle, (2)+asym_middle[3][1], color='red', linestyle='-.')
plot2.plot(angle, (0)+asym_middle[4][1], color='red', linestyle='-.')
plot2.plot(angle, (8)+asym_last[0][1], color='green')
plot2.plot(angle, (6)+asym_last[1][1], color='green')
plot2.plot(angle, (4)+asym_last[2][1], color='green')
plot2.plot(angle, (2)+asym_last[3][1], color='green')
plot2.plot(angle, (0)+asym_last[4][1], color='green')
plot2.set_title('') # change title here
plot2.set_xlabel(r'Θ$_{c.m.}$(deg)')
plot2.xaxis.set_minor_locator(MultipleLocator(10))
plot2.yaxis.set_minor_locator(MultipleLocator(1))
plot2.tick_params(which='major', length=8, width=2)
plot2.tick_params(which='minor', length=4, width=1)
plot2.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=False, labelright=False, direction='in')

plot3.plot(angle, (8)+asym_first[0][2], color='blue')
plot3.plot(angle, (6)+asym_first[1][2], color='blue')
plot3.plot(angle, (4)+asym_first[2][2], color='blue')
plot3.plot(angle, (2)+asym_first[3][2], color='blue')
plot3.plot(angle, (0)+asym_first[4][2], color='blue')
plot3.plot(angle, (8)+asym_middle[0][2], color='red', linestyle='-.')
plot3.plot(angle, (6)+asym_middle[1][2], color='red', linestyle='-.')
plot3.plot(angle, (4)+asym_middle[2][2], color='red', linestyle='-.')
plot3.plot(angle, (2)+asym_middle[3][2], color='red', linestyle='-.')
plot3.plot(angle, (0)+asym_middle[4][2], color='red', linestyle='-.')
plot3.plot(angle, (8)+asym_last[0][2], color='green')
plot3.plot(angle, (6)+asym_last[1][2], color='green')
plot3.plot(angle, (4)+asym_last[2][2], color='green')
plot3.plot(angle, (2)+asym_last[3][2], color='green')
plot3.plot(angle, (0)+asym_last[4][2], color='green')
plot3.set_title('') # change title here
plot3.set_xlabel(r'Θ$_{c.m.}$(deg)')
plot3.xaxis.set_minor_locator(MultipleLocator(10))
plot3.tick_params(which='major', length=8, width=2)
plot3.tick_params(which='minor', length=4, width=1)
plot3.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=False, labelright=False, direction='in')

# edit the line label locations here

plot1.text(12,8.4, levels[0]+' MeV', fontsize=7)
plot1.text(12,6.4, levels[1]+' MeV', fontsize=7)
plot1.text(12,4.6, levels[2]+' MeV', fontsize=7)
plot1.text(12,2.5, levels[3]+' MeV', fontsize=7)
plot1.text(8,.6, levels[4]+' MeV', fontsize=7)
plot2.text(12,8.4, levels[0]+' MeV', fontsize=7)
plot2.text(12,6.4, levels[1]+' MeV', fontsize=7)
plot2.text(12,4.6, levels[2]+' MeV', fontsize=7)
plot2.text(12,2.5, levels[3]+' MeV', fontsize=7)
plot2.text(8,.6, levels[4]+' MeV', fontsize=7)
plot3.text(12,8.4, levels[0]+' MeV', fontsize=7)
plot3.text(12,6.4, levels[1]+' MeV', fontsize=7)
plot3.text(12,4.6, levels[2]+' MeV', fontsize=7)
plot3.text(12,2.5, levels[3]+' MeV', fontsize=7)
plot3.text(8,.6, levels[4]+' MeV', fontsize=7)


plt.show()










