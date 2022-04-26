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
data = []
var_trig = []
chi=[]
final_index = 0
to_break=0
param_orig=''
with open(file1) as f:
    data = f.read().splitlines()
file2_mid = file2+'.mid'
param_file=file1.replace('_last.ecis','_param.txt')

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
for q in range(0,len(classic)):
    param[q]=hard_data[q][0]        
            
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
#Bunch of sys exit codes to break or not break loop
if  loop_num > 1:
    print('DONE with ', file2.partition('.inp')[0])
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

    with open(param_file) as g:
        lines=g.read().splitlines()
    orig_param = lines[-1]
    final_param = lines[1]
    g.close()
    final_var_str =''
    orig_var_str =''
    mag_orig = 0.0; mag_final = 0.0
    for i in range(len(classic)):
        orig_var = Decimal(orig_param.split()[i+2])*Decimal(1.0)
        final_var = Decimal(final_param.split()[i])*Decimal(1.0)
        mag_orig = mag_orig +float(orig_var)**2
        mag_final = mag_final + float(final_var)**2
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
    with open('results_final_fit.txt','a') as final:
        info = (chemical, shape, atomic, proton, neutron, energy_fl, betas, float(chi[len(chi)-1]), float(mid_chi),
                orig_var_str,final_var_str,mag_orig**0.5,mag_final**0.5)
        final.write("%5s %12s   %3d     %3d    %3d     %5.1f  %8.5f   %8.2f   %8.2f        %s        %s    %6.2f     %6.2f\n" %info)
    final.close()
    sys.exit(1)
