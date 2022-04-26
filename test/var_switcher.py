import sys
import os.path
from decimal import *
import numpy as np
file1 = sys.argv[1]
file2 = sys.argv[2]
loop_num = int(sys.argv[3])
chemical = sys.argv[4]
chemical_inp = sys.argv[5]
energy = sys.argv[6]
isospin = sys.argv[7]
version = sys.argv[8]
data = []
var_trig = []
chi=[]
final_index = 0
to_break=0
param_orig=''
with open(file1) as f:
    data = f.read().splitlines()
chi_file=file1.replace('.ecis','.txt')
var_file=file1.replace('_last.ecis','_var.txt')
file2_mid = file2+'.mid'
param_file=file1.replace('_last.ecis','_param.txt')
results_name = ''
if version == 'initial':
    results_name = 'results_original.txt'
if version == 'final':
    results_name = 'results_final.txt'

#********************************
#Make assignments here
#Specify parameter in ecis and respective line in potential code to be changed
 
hard_data = [[9,3],[13,4]]

#Finding number of variables to figure out number of arrays need to be created
with open(file2) as s:
    temp = s.read().splitlines()
classic=temp[len(temp)-2].split()

for l in range (0,len(temp)):
    if '   0.00000   0.00000   0.00000' in temp[l]:
        switch_trig=l-8

var =[[] for a in range(0,len(classic))]
param=[0.0]*len(classic)
#for q in range(0,len(classic)):
    #param[q]=hard_data[q][0]        
            
#Saving param trigs to an array
data_states = []
for i in range(0,len(data)):
    if '*** variables' in data[i]:
        var_trig.append(i)
    if '**** first control card ****' in data[i]:
        var_trig.append(i)
for j in range(0,len(var_trig)-1):
    var_line=data[var_trig[j]+2].split()
    for b in range(var_trig[j],var_trig[j+1]):
        if '***** chi2 =' in data[b]:
            chi_line=data[b].split()
    for e in range(0,len(var_line)-1):#removing index's in front of parameters in this loop
        if e%2 == 0:
            var_line[e]="False"
    try:
        while True:
            var_line.remove("False")
    except ValueError:
        pass
    for o in range(len(var_line)):#saving parameters to respective array
        var[o].append(var_line[o])
    chi.append(float(chi_line[8].replace('D','e')))


# check for first time through loop, if so get original paarameters
if loop_num==1:
    with open(file2_mid) as s_mid:
        temp_mid = s_mid.read().splitlines()
        with open(param_file,"w") as par:
            par.write('Original:\n')
            for i in range(0,len(classic)):
                switch_var_line=temp_mid[switch_trig+hard_data[i][1]]
                split=switch_var_line.split()
                param_orig=split[0]
                par.write("{:<12}".format(param_orig.replace('D','e')))
                par.write('  ')
            par.write('\n')
        par.close()     
    s_mid.close()

#Using hard data array to make required switches to the input file
CHECK_STR_ORIG=['']*len(classic)
CHECK_STR_NEW=['']*len(classic)
try_string=''
for i in range(0,len(classic)):
    switch_var_line=temp[switch_trig+hard_data[i][1]]
    split=switch_var_line.split()
    string=str("%.5f" % (float(var[i][len(var[i])-1].replace('D','e'))))
    CHECK_STR_ORIG[i]=split[0]
    CHECK_STR_NEW[i]=string
    new_string="{:>10.5f}{:>10.5f}{:>10.5f}".format(float(string),float(split[1]),float(split[2]))
    for m in range(0,len(temp)):
        if switch_var_line in temp[m]:
            temp[m]=new_string
    #Saving parameters to a file
    try_string+=CHECK_STR_ORIG[i]+" "
#Try to change Jump, reduce eachtime
switch_jump_line = temp[switch_trig+10]
split=switch_jump_line.split()
#if version == 'initial':
old_jump = float(split[-1])
#new_jump = max(5.0,old_jump-10.0)
new_jump_string = "{:>5d}{:>5d}{:>5d}{:>5d}{:>10.2f}".format(int(split[0]),int(split[1]),
                                                                    int(split[2]),int(split[3]),old_jump)
temp[switch_trig+10] = new_jump_string

print(try_string, file=open(var_file, "a"))


if version == 'initial':
    #Finding absolute values between all params
    def abs_cal():
        abs_value=0
        for i in range(0,len(classic)):
            orig=CHECK_STR_ORIG[i]
            new=CHECK_STR_NEW[i]
            abs_temp=abs(float(orig)-float(new))
            if abs_temp > abs_value:
                abs_value = abs_temp
        return abs_value 

    #Finding uniques in list
    def unique_for_x(lists):
        # intilize a null list 
        unique_list = [] 
        # traverse for all elements 
        for x in lists: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list

    #Update input file with new values
    def update(arr):
        print(arr[0], file=open(file2, "w"))
        for i in range(1,len(arr)):
            print(arr[i], file=open(file2, "a"))

    def percent_diff(num1,num2):
        numerator = abs(num1-num2)
        denominator = (num1 + num2)/2
        return (numerator/denominator)*100

    #Break loop in case all parameters in case they occur more than 3 times
    def should_i_break():
        global final_index
        to_break=0
        with open(var_file) as g:
            lines=g.read().splitlines()
        if len(lines) >= 3:
            #Decimal Context to keep only 3 decimals
            getcontext().rounding= ROUND_DOWN
            getcontext().prec=4
            var_from_list=[[] for a in range(0,len(classic))]
            uni_var = [[] for a in range(0,len(classic))]
            for index in lines:
                index_param = index.split()
                for a in range(0,len(classic)):
                    var_from_list[a].append(float(Decimal(index_param[a])*Decimal(1.0))) #rounds off
            index = [0.0]*len(classic)
            for a in range(0,len(classic)):
                uni_var[a]=unique_for_x(var_from_list[a])
                for element in uni_var[a]:
                    temp_count=var_from_list[a].count(element)
                    if temp_count >= 3:
                        index[a] = max(loc for loc, val in enumerate(var_from_list[a]) if val == element)
                        if(index[a] == index[0]):
                            final_index = index[a]
                            to_break+=1
                            break
        g.close()
        return to_break

    update(temp)
    absolute_true = abs_cal()

    #Saving params to a file for tailored values
    if loop_num!=1:
        for i in range(0,len(chi)):
            some_string="{:<15}".format(chi[i])
            for o in range(0,len(classic)):
                some_string+="{:<12}".format(float(var[o][i].replace("D","e")))
            some_string+="{:5}".format(loop_num)
            print(some_string, file=open(param_file, "a"))
    else:
        first_string="{:<6}{:<15}".format("First",chi[0])
        for e in range(0,len(classic)):
            first_string+="{:<12}".format(float(var[e][0].replace("D","e")))
        first_string+="{:5}".format(loop_num)
        print(first_string, file=open(param_file, "a"))
        for t in range(1,len(chi)):
            second_string="{:<15}".format(chi[t])
            for p in range(0,len(classic)):
                second_string+="{:<12}".format(float(var[p][t].replace("D","e")))
            second_string+="{:5}".format(loop_num)
            print(second_string, file=open(param_file, "a"))


    count1=should_i_break()
    if count1 == len(classic):
        to_break=1

if version == 'final':
    first_string="{:<6}{:<15}".format("First",chi[0])
    for e in range(0,len(classic)):
        first_string+="{:<12}".format(float(var[e][0].replace("D","e")))
    first_string+="{:5}".format(loop_num)
    print(first_string, file=open(param_file, "a"))
    for t in range(1,len(chi)):
        second_string="{:<15}".format(chi[t])
        for p in range(0,len(classic)):
            second_string+="{:<12}".format(float(var[p][t].replace("D","e")))
        second_string+="{:5}".format(loop_num)
        print(second_string, file=open(param_file, "a"))
    print('DONE with ', file2.partition('.inp')[0], ' on loop number#',loop_num)
    print(chi[len(chi)-1], file=open(chi_file, "w"))
    atomic=0
    proton=0
    neutron=0
    shape = ''
    betas=0.0
    energy_fl =  float(energy)
    with open(chemical_inp,"r") as ch:
        data = ch.read().splitlines()
        atomic = int(data[4])
        proton = int(data[3])
        neutron = atomic - proton
        shape = data[5]
    ch.close()
    with open(file1.replace('_last.ecis','.betas')) as beta:
        data = beta.read()
        betas = float(data)
    beta.close()  
    getcontext().rounding= ROUND_DOWN
    getcontext().prec=4

    with open(param_file) as h:
        param_data=h.read().splitlines()
    orig_param = param_data[1]
    h.close()
    with open(var_file) as g:
        lines=g.read().splitlines()
    g.close()
    var_from_list=[[] for a in range(0,len(classic))]
    for index in lines:
       index_param = index.split()
       for a in range(0,len(classic)):
           var_from_list[a].append(float(Decimal(index_param[a])*Decimal(1.0))) #rounds off
    if final_index==0:
        final_index= len(lines)-1
    final_var_str =''
    orig_var_str =''
    mag_orig = 0.0;
    mag_final = 0.0
    for i in range(len(classic)):
        final_var = float(var_from_list[i][final_index])
        orig_var = Decimal(orig_param.split()[i])*Decimal(1.0)
        #final_var = Decimal(final_var)*Decimal(1.0)
        mag_orig = mag_orig + float(orig_var) ** 2
        mag_final = mag_final + float(final_var) ** 2
        pad ='   '
        if final_var < 0:
            pad = '  '
        final_var_str+= str(final_var)+pad
        if orig_var > 0:
            pad = '   '
        orig_var_str+= str(orig_var)+pad    
    # get first chi^2 value
    mid_chi = ''
    reyna =[]
    mid_file = file1.replace('last','middle')
    with open(mid_file) as v:
        reyna = v.read().splitlines()
    v.close()    
    for i in range(0, len(reyna)):
        if ' run   1   max =   1   ***** chi2 =' in reyna[i]:
            mid_chi = reyna[i][37:53].replace('D','e') 
            break
    with open(results_name,'a') as final: 
        info = (chemical, shape, atomic, proton, neutron, isospin, energy_fl, betas,
                float(chi[len(chi)-1]), float(mid_chi), final_var_str,orig_var_str, mag_final**0.5,mag_orig**0.5)
        final.write("%5s %12s   %3d   %3d   %3d   %1s  %5.1f   %8.5f   %8.2f   %8.2f     %s     %s      %6.2f   %6.2f\n" %info)
    final.close()
    sys.exit(1)

#Bunch of sys exit codes to break or not break loop
if (absolute_true < 0.01 and int(loop_num >3)) or to_break == 1 or str(loop_num) == '30':
    print('DONE with ', file2.partition('.inp')[0], ' on loop number#',loop_num)
    print(chi[len(chi)-1], file=open(chi_file, "w"))
    atomic=0
    proton=0
    neutron=0
    shape = ''
    betas=0.0
    energy_fl =  float(energy)
    with open(chemical_inp,"r") as ch:
        data = ch.read().splitlines()
        atomic = int(data[4])
        proton = int(data[3])
        neutron = atomic - proton
        shape = data[5]
    ch.close()
    with open(file1.replace('_last.ecis','.betas')) as beta:
        data = beta.read()
        betas = float(data)
    beta.close()  
    getcontext().rounding= ROUND_DOWN
    getcontext().prec=4

    with open(param_file) as h:
        param_data=h.read().splitlines()
    orig_param = param_data[1]
    h.close()
    with open(var_file) as g:
        lines=g.read().splitlines()
    g.close()
    var_from_list=[[] for a in range(0,len(classic))]
    for index in lines:
       index_param = index.split()
       for a in range(0,len(classic)):
           var_from_list[a].append(float(Decimal(index_param[a])*Decimal(1.0))) #rounds off
    if final_index==0:
        final_index= len(lines)-1
    final_var_str =''
    orig_var_str =''
    mag_orig = 0.0;
    mag_final = 0.0
    last_param_var = param_data[len(param_data)-1].split()
    for i in range(len(classic)):
        #final_var = var_from_list[i][final_index]
        final_var = float(last_param_var[i+1])
        orig_var = Decimal(orig_param.split()[i])*Decimal(1.0)
        #final_var = Decimal(final_var)*Decimal(1.0)
        mag_orig = mag_orig + float(orig_var) ** 2
        mag_final = mag_final + float(final_var) ** 2
        pad ='   '
        if final_var < 0:
            pad = '  '
        final_var_str+= str(final_var)+pad
        if orig_var > 0:
            pad = '   '
        orig_var_str+= str(orig_var)+pad    
    # get first chi^2 value
    mid_chi = ''
    reyna =[]
    mid_file = file1.replace('last','middle')
    with open(mid_file) as v:
        reyna = v.read().splitlines()
    v.close()    
    for i in range(0, len(reyna)):
        if ' run   1   max =   1   ***** chi2 =' in reyna[i]:
            mid_chi = reyna[i][37:53].replace('D','e') 
            break
    with open(results_name,'a') as final: 
        info = (chemical, shape, atomic, proton, neutron, isospin, energy_fl, betas,
                float(chi[len(chi)-1]), float(mid_chi), final_var_str,orig_var_str, mag_final**0.5,mag_orig**0.5)
        final.write("%5s %12s   %3d   %3d   %3d   %1s  %5.1f   %8.5f   %8.2f   %8.2f     %s     %s      %6.2f   %6.2f\n" %info)
    final.close()
    sys.exit(1)
