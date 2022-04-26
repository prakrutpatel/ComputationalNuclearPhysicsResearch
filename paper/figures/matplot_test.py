import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# uploading the data from file and splitting it into arrays, needed to be converted to float in order to be plotted

with open("plot_numbers6_2.txt") as f:
        data = f.read().splitlines()

angle = []
crosssection = []
csruther = []
polarization = []
for i in range(1, len(data)):
    line = data[i].split()
    angle.append(float(line[0]))
    crosssection.append(float(line[1]))
    csruther.append(float(line[2]))
    polarization.append(float(line[3]))
 
# creating the plot figures here and making them similar to the publication figures
# potentially update x_axis.set_major_locator to change tick frequency on x axis

fig = plt.figure(figsize=(9.6, 5.6))
plot1 = fig.add_subplot(131)
plot2 = fig.add_subplot(132)
plot3 = fig.add_subplot(133)
plt.subplots_adjust(wspace=.04, hspace=.04)

plot1.plot(angle, crosssection, color='red')
plot1.set_title('1')
plot1.set_xlabel(r'$Θ_{c.m.}$(deg)')
plot1.set_ylabel('Cross-Section')
plot1.set_yscale('log')
plot1.xaxis.set_minor_locator(MultipleLocator(10))
plot1.tick_params(which='major', length=8, width=2)
plot1.tick_params(which='minor', length=4, width=1)
plot1.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True, 
                  labeltop=False, labelleft=True, labelright=False, direction='in')

plot2.plot(angle, crosssection, color='green')
plot2.set_title('2')
plot2.set_xlabel(r'$Θ_{c.m.}$(deg)')
plot2.set_yscale('log')
plot2.xaxis.set_minor_locator(MultipleLocator(10))
plot2.tick_params(which='major', length=8, width=2)
plot2.tick_params(which='minor', length=4, width=1)
plot2.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=False, labelright=False, direction='in')

plot3.plot(angle, crosssection, color='blue')
plot3.set_title('3')
plot3.set_xlabel(r'$Θ_{c.m.}$(deg)')
plot3.set_yscale('log')
plot3.xaxis.set_minor_locator(MultipleLocator(10))
plot3.tick_params(which='major', length=8, width=2)
plot3.tick_params(which='minor', length=4, width=1)
plot3.tick_params(which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                  labeltop=False, labelleft=False, labelright=False, direction='in')


plt.show()

