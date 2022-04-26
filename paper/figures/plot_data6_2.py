import numpy as np
import matplotlib.pyplot as plt

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
 
# creating the plot figures here

fig = plt.figure(figsize=(14, 7))
plot1 = fig.add_subplot(141)
plot2 = fig.add_subplot(142)
plot3 = fig.add_subplot(143)
plot4 = fig.add_subplot(144)

plot1.plot(angle, crosssection, color='red')
plot1.set_xlabel('Angle [Deg.]')
plot1.set_ylabel('Cross-Section')
plot1.set_title('Angle vs. Cross-Section')

plot2.plot(angle, crosssection, color='red')
plot2.set_xlabel('Angle [Deg.]')
plot2.set_yscale('log')
plot2.set_ylabel('Cross-Section (Log Scale)')
plot2.set_title('Angle vs. Cross-Section (Log Scaled)')

plot3.plot(angle, csruther, color='green')
plot3.plot(angle, polarization, color='blue')
plot3.set_xlabel('Angle [Deg.]')
plot3.set_ylabel('c.s./ruther.')
plot3.set_title('Angle vs. c. s./ruther.')

plot4.plot(angle, polarization, color='blue')
plot4.set_xlabel('Angle [Deg.]')
plot4.set_ylabel('Polarization')
plot4.set_title('Angle vs. Polarization')

# additional cosmetics on the plots


plt.show()

