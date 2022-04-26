import numpy as np
from numpy import vectorize, angle
from decimal import *
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from os import path
import pandas as pd
import os
plt.rcParams.update({'font.size': 50})
plt.rcParams.update({'errorbar.capsize': 10})
plt.rcParams.update({'savefig.format': 'eps'})
plotdata = sys.argv[1]
isospin = sys.argv[2]
version = sys.argv[3]
isospin_label ='proton'
if isospin =='n':
    isospin_label = 'neutron'
trigger = []
trigger_dub = []
trigger_dub1 = []
new_string = []
new_string_dub = []
new_string_dub1 = []
angles = []
cs = []
polar = []
data_states = []
data_states_dub = []
data_states_dub1 = []
state_data=[]
state_data_dub1=[]
raze=[]
mid_chi=0
unique_element_descriptor=plotdata.replace('.ecis','')
file_case = str('first' in plotdata)
print("Printing ",file_case,plotdata)

CHECK_LAST=str('last' in plotdata)
alpha = 0.00729735
hc = 197.327
Z1 = 1
# Splitting incoming data into array of strings
with open(plotdata) as f:
    data = f.read().splitlines()

# Getting info for plot title
n = data[65].split()
nucluei=n[1]
proj_energy=n[2]
proj_unit=n[3]

# Searching for trigger word for initial angle, step size and final angle

for i in range(0, len(data)):
    if 'scattering angles from' in data[i]:
        scatter_trig = i
    if 'excitation energy' in data[i]:
        state_data.append(i)
    if 'product of charges' in data[i]:
        pc = i
    if 'energy(c. m.)' in data[i]:
        cm = i
# Initializing angles and step size but splitting a string in smaller strings
line_scatter = data[scatter_trig]
new_line_scatter = line_scatter.split()
initial = int(float(new_line_scatter[3]))
step = int(float(new_line_scatter[7]))
to = int(float(new_line_scatter[10]))
Ztar_line = data[pc].split()
Ztar = float(Ztar_line[8])
Ek_line = data[cm].split()
Ek = float(Ek_line[7].replace('D','e'))


def cs_with_rf(ang,data):
    newList = [(np.asarray(10.0)*np.power((Z1*Ztar*alpha*hc)/(4*np.asarray(Ek)*np.power(np.sin(x*np.pi/360),2)),2)) for x in ang]
#    ang_np = np.asarray(ang, dtype = 'float64')
#    sin_np = np.power(np.sin(ang_np*np.pi/360.0),-4)
#    newList = (10.0*np.power((Z1*Ztar*alpha*hc)/(4.0*Ek*np.power(np.sin(ang_np*np.pi/360),2)),2))
    return[float(b) / float(m) for b,m in zip(data, newList)]

# figure out math required for saving data for each state
if (initial % 2) == 0:
    math = int(((to - initial) / step) + 2)
if (initial % 2) != 0:
    math = int(((to - initial) / step) + 3)

#Using file case value to figure out when to plot and when to save cs, polar and ang values to a file to be used in off case in potential.py. Create plot when last is in filename and create ang,cs and polar files when first is in the name

if file_case == "True":
# Searching for trigger word to know starting point for each state
    for i in range(0, len(data)):
        if 'asym. or it11' in data[i]:
            trigger.append(i)
        if 'total reaction' in data[i]:
            rcs = float((data[i].split())[5])
        if 'total cross section' in data[i]:
            tcs = float((data[i].split())[5])
else:
    for i in range(0, len(data)):
        if ' elastic scattering on the target state of spin' in data[i]:
            elastic_trig = i

    for i in range (elastic_trig, len(data)):
        if 'asym. or it11' in data[i]:
            trigger.append(i)

numpy_array = [np.array(np.empty(0),dtype=float) for i in range (len(trigger))]

no_states = len(trigger)
for i in range(0, no_states):
    #need to improve the 180 inflexability
    temp_arr = data[trigger[i]+1:trigger[i]+math]
    data_states.append(temp_arr)
#counters to keep track of state and line
index = 0
for elements in data_states:
    index1=0
    for element_line in elements:
        #convert D format for double precision in fortran to e format(scientific format)
        new_string = element_line.replace('D','e')
        temp = np.fromstring(new_string,dtype=float,sep=" ")
        skip = len(temp)
        # It looked like the easiest to understand was to insert as
        # a long array then reshape later
        numpy_array[index]=np.insert(numpy_array[index],[index1*skip],temp)
        index1 += 1
    # at the end I reshape each state
    numpy_array[index] = np.reshape(numpy_array[index],(index1,skip))
    index += 1
if CHECK_LAST == "True":
    if version =='initial':
        mid_file = plotdata.replace('last','middle')
        with open(mid_file) as v:
            reyna = v.read().splitlines()
        for i in range(0, len(reyna)):
            if 'scattering angles from' in reyna[i]:
                scatter_trig_dub1 = i
            if 'excitation energy' in reyna[i]:
                state_data_dub1.append(i)
            if 'product of charges' in reyna[i]:
                pc_dub1 = i
            if 'energy(c. m.)' in reyna[i]:
                cm_dub1 = i
        line_scatter_dub1 = reyna[scatter_trig_dub1]
        new_line_scatter_dub1 = line_scatter_dub1.split()
        initial_dub1 = int(float(new_line_scatter_dub1[3]))
        step_dub1 = int(float(new_line_scatter_dub1[7]))
        to_dub1 = int(float(new_line_scatter_dub1[10]))
        if (initial_dub1 % 2) == 0:
            math_dub1 = int(((to_dub1 - initial_dub1) / step_dub1) + 2)
        if (initial_dub1 % 2) != 0:
            math_dub1 = int(((to_dub1 - initial_dub1) / step_dub1) + 3)
        for i in range(0, len(reyna)):
            if ' elastic scattering on the target state of spin' in reyna[i]:
                elastic_trig_dub1 = i
            if ' run   1   max =   1   ***** chi2 =' in reyna[i]:
                raze.append(i)

        for i in range (elastic_trig_dub1, len(reyna)):
            if 'asym. or it11' in reyna[i]:
                trigger_dub1.append(i)

        numpy_array_dub1 = [np.array(np.empty(0),dtype=float) for i in range (len(trigger_dub1))]  
        no_states_dub1 = len(trigger_dub1)
        for i in range(0, no_states_dub1):
            temp_arr_dub1 = reyna[trigger_dub1[i]+1:trigger_dub1[i]+math_dub1]
            data_states_dub1.append(temp_arr_dub1)
        #counters to keep track of state and line
        index_dub1 = 0
        for elements in data_states_dub1:
            index1_dub1=0
            for element_line in elements:
                #convert D format for double precision in fortran to e format(scientific format)
                new_string_dub1 = element_line.replace('D','e')
                temp_dub1 = np.fromstring(new_string_dub1,dtype=float,sep=" ")
                skip_dub1 = len(temp_dub1)
                # It looked like the easiest to understand was to insert as
                # a long array then reshape later
                numpy_array_dub1[index_dub1]=np.insert(numpy_array_dub1[index_dub1],[index1_dub1*skip_dub1],temp_dub1)
                index1_dub1 += 1
            # at the end I reshape each state
            numpy_array_dub1[index_dub1] = np.reshape(numpy_array_dub1[index_dub1],(index1_dub1,skip_dub1))
            index_dub1 += 1
        sage=reyna[raze[0]].split()
        mid_chi=float(sage[8].replace('D','e'))
        
#saving values to a file to be used again
if file_case == "True":
    isospin_dimension=3
    if isospin =='n':
        isospin_dimension=2
    for i in range(0,len(numpy_array[0])):
        angles.insert(i,numpy_array[0][i][0])
        cs.insert(i,numpy_array[0][i][1])
        polar.insert(i,numpy_array[0][i][isospin_dimension])
        np.save(nucluei+str(proj_energy)+isospin+'angles.npy',angles)
        np.save(nucluei+str(proj_energy)+isospin+'cs.npy',cs)
        np.save(nucluei+str(proj_energy)+isospin+'polar.npy',polar)
    np.save(nucluei+str(proj_energy)+isospin+'rcs.npy',rcs)
    if isospin == 'n':
        np.save(nucluei+str(proj_energy)+isospin+'tcs.npy',tcs)


# this is only state 0
if file_case == "False":
    init_file = plotdata.replace('last','first')

    with open(init_file) as f:
        lines = f.read().splitlines()

    for i in range(0, len(lines)):
        if 'asym. or it11' in lines[i]:
            trigger_dub.append(i)


    for i in range(0, len(lines)):
        if 'scattering angles from' in lines[i]:
            scatter_trig_dub = i


    line_scatter_dub = lines[scatter_trig_dub]
    new_line_scatter_dub = line_scatter_dub.split()
    initial_dub = int(float(new_line_scatter_dub[3]))
    step_dub = int(float(new_line_scatter_dub[7]))
    to_dub = int(float(new_line_scatter_dub[10]))

    if (initial_dub % 2) == 0:
        math_dub = int(((to_dub - initial_dub) / step_dub) + 2)
    if (initial_dub % 2) != 0:
        math_dub = int(((to_dub - initial_dub) / step_dub) + 3)

    numpy_array_dub = [np.array(np.empty(0),dtype=float) for i in range (len(trigger_dub))]
    no_states_dub = len(trigger_dub)


    for i in range(0, no_states_dub):
        temp_arr_dub = lines[trigger_dub[i]+1:trigger_dub[i]+math_dub]
        data_states_dub.append(temp_arr_dub)
    index_dub = 0
    for elements in data_states_dub:
        index1_dub = 0
        for element_line in elements:
            new_string_dub = element_line.replace('D','e')
            temp_dub = np.fromstring(new_string_dub,dtype=float,sep=" ")
            skip_dub = len(temp_dub)
            numpy_array_dub[index_dub]=np.insert(numpy_array_dub[index_dub],[index1_dub*skip_dub],temp_dub)
            index1_dub += 1
        numpy_array_dub[index_dub] = np.reshape(numpy_array_dub[index_dub],(index1_dub,skip_dub))
        index_dub += 1

# saving data into numpy array
# note that empty is key to starting with nothing, decided to go with 1-D array
# then reshape later

if file_case == "False":
    #To save png file for each state
    for p in range(0, len(trigger)):
        per_trig=0
        per_trig_pol=0
        dim_trig=0
        dim_trig_pol=0
        no_dim=0
        no_dim_pol=0
        polang=[]
        poldata=[]
        polerr=[]
        expang = []
        dataexp=[]
        experr=[]
        temp_data=[]
        data_trig=[]
        data_trig_pol=[]
        ang_col=0
        ang_col_pol=0
        dat_col=0
        dat_col_pol=0
        err_col=-1
        err_col_pol=-1
        endcommon_trig=[]
        endcommon_trig_pol=[]
        name_holder=[]
        name_holder_pol=[]
        temp_string = data[state_data[p]].split()
        expdata = nucluei+str(proj_energy)+'State'+str(p)+'.dat'
        polexpdata = nucluei+str(proj_energy)+'State'+str(p)+'pol'+'.dat'
        # special case for neutron scattering
        if isospin =='n':
            expdata = nucluei + str(proj_energy) + 'nState' + str(p) + '.dat'
            polexpdata = nucluei + str(proj_energy) + 'nState' + str(p) + 'pol' + '.dat'
        chi_data=plotdata.replace('.ecis','.txt')
        with open(chi_data) as b:
            chi_sqr_list = b.read().splitlines() 
        chi_sqr_true = float(chi_sqr_list[0])
        chi_sqr=("%.2f" % chi_sqr_true)
        expstr = str(path.exists(expdata))
        polexpstr = str(path.exists(polexpdata))
        
        #This block is used to import experimental data and store them for plotting purposes
        if expstr == "True":
            with open(expdata) as f:
                data1 = f.read().splitlines()
            for i in range(0, len(data1)):
                if (('ENDCOMMON' in data1[i]) or ('NOCOMMON' in data1[i])):
                    endcommon_trig.append(i)
                if 'ANG-CM' in data1[i]:
                    line_ang = i
                if 'ENDDATA' in data1[i]:
                    dat_trig=i
            for m in range(endcommon_trig[len(endcommon_trig)-1], len(data1)):
                if 'DATA' in data1[m]:
                    data_trig.append(m)

            line_data = data1[data_trig[0]]
            new_line_data = line_data.split()
            no_of_columns = int(new_line_data[1])
            row_start = int(new_line_data[2])
            line_column = data1[line_ang]
            new_line_column = line_column.split()
            per_or_dim_line = data1[line_ang+1]
            new_per_or_dim_line = per_or_dim_line.split()

            for q in range(0,no_of_columns):
                name_holder.extend(str(q))


            for o in range(0,no_of_columns):
                if 'ANG' in new_line_column[o]:
                    ang_col=o
                if new_line_column[o] == 'DATA' or new_line_column[o] == 'DATA-CM':
                    dat_col=o
                if 'ERR' in new_line_column[o]:
                    err_col=o
                    
            for j in range(0,no_of_columns):
                if err_col != -1:
                    if 'PER' in new_per_or_dim_line[j] or 'PER-CENT' in new_per_or_dim_line[j]:
                        if j == err_col:
                            per_trig=1
                    if 'DIM' in new_per_or_dim_line[j] or 'NO-DIM' in new_per_or_dim_line[j] or 'DIMM' in new_per_or_dim_line[j] or 'MB/SR' in new_per_or_dim_line[j]:
                        if j == err_col:
                            dim_trig=1
            for j in range(0,no_of_columns):
                if 'NO-DIM' in new_per_or_dim_line[j] or 'NO-DIMM' in new_per_or_dim_line[j]:
                    if j == dat_col:
                        no_dim=1
            for i in range(dat_trig-row_start,dat_trig):
                print(data1[i], file=open(unique_element_descriptor+"panda_dump"+str(p)+".csv", "a"))

            pd_arr = pd.read_csv(unique_element_descriptor+"panda_dump"+str(p)+".csv", delim_whitespace = True, header = None,float_precision='round_trip', names=name_holder).fillna(0).values
            for i in range(0, len(pd_arr)):
                expang.append(pd_arr[i][ang_col])
                dataexp.append(pd_arr[i][dat_col])
                if err_col != -1:
                    if per_trig == 1:
                        experr.append(int((pd_arr[i][err_col])*(pd_arr[i][dat_col]))/100)
                    if dim_trig == 1:
                        experr.append(pd_arr[i][err_col])
                else:
                    experr.append(0)
            os.remove(unique_element_descriptor+"panda_dump"+str(p)+".csv")
        if polexpstr == "True":
            with open(polexpdata) as f:
                data2 = f.read().splitlines()
            for i in range(0, len(data2)):
                if (('ENDCOMMON' in data2[i]) or ('NOCOMMON' in data2[i])):
                    endcommon_trig_pol.append(i)
                if 'ANG-CM' in data2[i]:
                    line_ang_pol = i
                if 'ENDDATA' in data2[i]:
                    dat_trig_pol=i
            for m in range(endcommon_trig_pol[len(endcommon_trig_pol)-1], len(data2)):
                if 'DATA' in data2[m]:
                    data_trig_pol.append(m)

            line_data_pol = data2[data_trig_pol[0]]
            new_line_data_pol = line_data_pol.split()
            no_of_columns_pol = int(new_line_data_pol[1])
            row_start_pol = int(new_line_data_pol[2])
            line_column_pol = data2[line_ang_pol]
            new_line_column_pol = line_column_pol.split()
            per_or_dim_line_pol = data2[line_ang_pol+1]
            new_per_or_dim_line_pol = per_or_dim_line_pol.split()

            for q in range(0,no_of_columns_pol):
                name_holder_pol.extend(str(q))


            for o in range(0,no_of_columns_pol):
                if 'ANG' in new_line_column_pol[o]:
                    ang_col_pol=o
                if new_line_column_pol[o] == 'DATA' or new_line_column_pol[o] == 'DATA-CM':
                    dat_col_pol=o
                if 'ERR' in new_line_column_pol[o]:
                    err_col_pol=o
                    
            for j in range(0,no_of_columns_pol):
                if err_col_pol != -1:
                    if 'PER' in new_per_or_dim_line_pol[j] or 'PER-CENT' in new_per_or_dim_line_pol[j]:
                        if j == err_col_pol:
                            per_trig_pol=1
                    if 'DIM' in new_per_or_dim_line_pol[j] or 'NO-DIM' in new_per_or_dim_line_pol[j] or 'DIMM' in new_per_or_dim_line_pol[j] or 'MB/SR' in new_per_or_dim_line_pol[j]:
                        if j == err_col_pol:
                            dim_trig_pol=1
            for j in range(0,no_of_columns_pol):
                if 'NO-DIM' in new_per_or_dim_line_pol[j] or 'NO-DIMM' in new_per_or_dim_line_pol[j]:
                    if j == dat_col_pol:
                        no_dim_pol=1
            for i in range(dat_trig_pol-row_start_pol,dat_trig_pol):
                print(data2[i], file=open(unique_element_descriptor+"panda_dump_pol"+str(p)+".csv", "a"))

            pd_arr_pol = pd.read_csv(unique_element_descriptor+"panda_dump_pol"+str(p)+".csv", delim_whitespace = True, header = None,float_precision='round_trip', names=name_holder_pol).fillna(0).values
            for i in range(0, len(pd_arr_pol)):
                polang.append(pd_arr_pol[i][ang_col_pol])
                poldata.append(pd_arr_pol[i][dat_col_pol])
                if err_col_pol != -1:
                    if per_trig_pol == 1:
                        polerr.append(int((pd_arr_pol[i][err_col_pol])*(pd_arr_pol[i][dat_col_pol]))/100)
                    if dim_trig_pol == 1:
                        polerr.append(pd_arr_pol[i][err_col_pol])
                else:
                    polerr.append(0)
            os.remove(unique_element_descriptor+"panda_dump_pol"+str(p)+".csv")
        if ((len(numpy_array[p][0]) >2) and (len(numpy_array[p][0])  < 5)):
            # Now for protons and neutrons, may have three or four dims
            filename = nucluei+str(proj_energy)+isospin+'State'+str(p)+'.eps'
            if len(numpy_array[p][0]) == 4:
                ang0, cs0, r0, ay0 = np.split(numpy_array_dub[0],4,1)
                ang0_f, cs0_f, r0_f, ay0_f = np.split(numpy_array[0],4,1)
                if version =='initial':
                    ang0_f1, cs0_f1, r0_f1, ay0_f1 = np.split(numpy_array_dub1[0],4,1)
            if len(numpy_array[p][0]) == 3:
                ang0, cs0, ay0 = np.split(numpy_array_dub[0],3,1)
                ang0_f, cs0_f, ay0_f = np.split(numpy_array[0],3,1)
                if version =='initial':
                    ang0_f1, cs0_f1, ay0_f1 = np.split(numpy_array_dub1[0],3,1)
            plt.figure(p,dpi = 50,figsize=[36,24])
            fig = plt.subplot(121)
            if no_dim == 1:
                plt.errorbar(expang,dataexp,yerr=experr,fmt='o', markersize=8.0, markeredgewidth=10.0,label='NNDC Data with Error-Bar(Cross-section)',elinewidth=3)
            if no_dim == 0:
                plt.errorbar(expang,cs_with_rf(expang,dataexp),yerr=cs_with_rf(expang,experr),fmt='o', markersize=8, markeredgewidth=10,label='NNDC Data with Error-Bar(Cross-section)',elinewidth=3)
            if isospin =='p':
                fig.plot(ang0,cs_with_rf(ang0,cs0),'blue',label='Original DWBA',linewidth=10.0)
                if version =='initial':
                    fig.plot(ang0_f1,cs_with_rf(ang0_f1,cs0_f1),'green',label='Original DWBA w Coupled Channel',linewidth=10.0)
                fig.plot(ang0_f,cs_with_rf(ang0_f,cs0_f),'magenta',label='Distorted DWBA w Coupled Channel',linewidth=10.0)
            else:
                fig.plot(ang0, cs0, 'blue', label='Original DWBA', linewidth=10.0)
                if version =='initial':
                    fig.plot(ang0_f1, cs0_f1, 'green', label='Original DWBA w Coupled Channel',
                         linewidth=10.0)
                fig.plot(ang0_f, cs0_f, 'magenta', label='Distorted DWBA w Coupled Channel',
                         linewidth=10.0)
            fig.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0),ncol=1, fancybox=True, prop={'size': 20})
            fig.title.set_text('Elastic differential cross-section')
            plt.annotate(r'$\chi^{2}=$'+str(chi_sqr), xy=(0.01, 0.97), xycoords='axes fraction')
            plt.annotate(r'$first\:\chi^{2}=$'+str(mid_chi), xy=(0.01, 0.94), xycoords='axes fraction')
            plt.suptitle(nucluei+' ' + proj_energy+' ' +proj_unit+' '+isospin_label+' '
                         +', State: '+temp_string[6]+', Energy level: '+"%.2f" % float(temp_string[10])+' '+temp_string[11], fontsize = 90,y=0.95)
            fig.set_xlim(8,176)
            fig.set_xlabel('Angle C.M. [deg]', size='x-large')
            fig.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            fig.set_yscale('log')
            fig.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ ($\frac{mb}{sr}$)', size='x-large')
            fig = plt.subplot(122)
            if no_dim_pol == 1:
                plt.errorbar(polang,poldata,yerr=polerr,fmt='o', markersize=8.0, markeredgewidth=10.0,label='NNDC Data with Error-Bar(Polarization)',elinewidth=3)
            if no_dim_pol == 0:
                plt.errorbar(polang,cs_with_rf(polang,poldata),yerr=cs_with_rf(polang, polerr),fmt='o', markersize=8, markeredgewidth=10,label='NNDC Data with Error-Bar(Polarization)',elinewidth=3)
            fig.plot(ang0,ay0,'blue',label='Original DWBA',linewidth=10.0)
            if version =='initial':
                fig.plot(ang0_f1,ay0_f1,'green',label='Original DWBA w Coupled Channel',linewidth=10.0)
            fig.plot(ang0_f,ay0_f,'magenta',label='Distorted DWBA w Coupled Channel',linewidth=10.0)
            fig.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0),ncol=1, fancybox=True, prop={'size': 20})
            fig.title.set_text('Polarization')
            fig.set_xlim(8,176)
            fig.set_xlabel('Angle C.M. [deg]', size='x-large')
            fig.set_ylabel(r'$A_{y}$', size='x-large')
            plt.savefig(filename,dpi=50, transparent = False)
            plt.close()
        else:
            ang1, cs1, ay1, vp1, sf1 = np.split(numpy_array_dub[p],5,1)
            ang1_f, cs1_f, ay1_f, vp1_f, sf1_f = np.split(numpy_array[p],5,1)
            filename = nucluei+str(proj_energy)+isospin+'State'+str(p)+'.eps'
            plt.figure(p,dpi = 50,figsize=[36,24])
            fig = plt.subplot(121)
            plt.errorbar(expang,dataexp,yerr=experr,fmt='o', markersize=8, markeredgewidth=10,label='NNDC Data with Error-Bar(Cross-section)',elinewidth=3)
            fig.plot(ang1,cs1,'blue',label='Original DWBA',linewidth=10.0)
            fig.plot(ang1_f,cs1_f,'magenta',label='Distorted DWBA w Coupled Channel',linewidth=10.0)
            fig.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0),ncol=1, fancybox=True, prop={'size': 20})
            fig.title.set_text('Inelastic differential cross-section')
            plt.annotate(r'$\chi^{2}b=$'+str(chi_sqr), xy=(0.01, 0.97), xycoords='axes fraction')
            plt.annotate(r'$first\:\chi^{2}=$'+str(mid_chi), xy=(0.01, 0.94), xycoords='axes fraction')
            plt.suptitle(nucluei+' ' + proj_energy+' ' +proj_unit+', State: '+temp_string[6]+', Energy level: '+"%.2f" % float(temp_string[10])+' '+temp_string[11], fontsize = 90,y=0.95)
            fig.set_xlim(8,176)
            fig.set_xlabel('Angle C.M. [deg]', size='x-large')
            fig.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d")) 
            fig.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ ($\frac{mb}{sr}$)', size='x-large')
            fig = plt.subplot(122)
            plt.errorbar(polang,poldata,yerr=polerr,fmt='o', markersize=8.0, markeredgewidth=10.0,label='NNDC Data with Error-Bar(Polarization)',elinewidth=3)
            fig.plot(ang1,ay1,'blue',label='Original DWBA',linewidth=10.0)
            fig.plot(ang1_f,ay1_f,'magenta',label='Distorted DWBA w Coupled Channel',linewidth=10.0)
            fig.legend(loc='upper right',bbox_to_anchor=(1.0, 1.0),ncol=1, fancybox=True, prop={'size': 20})
            fig.title.set_text('Polarization')
            fig.set_xlim(8,176)
            fig.set_xlabel('Angle C.M. [deg]', size='x-large')
            fig.set_ylabel(r'$A_{y}$', size='x-large')
            plt.savefig(filename,dpi=50, transparent = False)
            plt.close()
            del ang1, cs1, ay1, vp1, sf1
            del ang1_f, cs1_f, ay1_f, vp1_f, sf1_f

    del expang,dataexp,experr,temp_data,data_trig
