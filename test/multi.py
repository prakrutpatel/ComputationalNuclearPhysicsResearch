
import multiprocessing
from multiprocessing import pool
import sys
from subprocess import call
import time
import os
import re

def remove(string): #some regex to remove spaces
    pattern = re.compile(r'\s+')
    return re.sub(pattern, '', string)

found_ecis = False
first_time = True
# Compile ecis if it does not exist
while not found_ecis:
    if os.path.exists('ecis'):
        print ('Found Ecis ...')
        found_ecis = True
    else:
        if first_time:
            first_time = False
            os.system('gfortran -o ecis -O3 ecis_tallys_06.f')
        time.sleep(1)
        print('Compiling Ecis .....')

temp_data = []
data2 =[]
no_thread = multiprocessing.cpu_count()
input_file = sys.argv[1]
version = sys.argv[2]
if version == 'initial':
    results_name = 'results_original.txt'
if version == 'final':
    results_name = 'results_final.txt'
# reading in of input file andextensive checking to make sure of integrity
with open(input_file) as f:
    data = f.read().splitlines()
found_error = False
found_end = False
for i in range(0,len(data)):
    if 'end' in data[i]:
        end = i
        found_end = True
    else:
        label_isospin =data[i].split()[-1]
        data2.append(label_isospin)
        if ((label_isospin != 'pro') and (label_isospin != 'neu')):
            found_error = True
        data[i] = data[i].split()[0] + '  ' + data[i].split()[1]
        if (len(data[i].split()) != 2):
            print(len(data[i].split()))
            found_error = True
        if (data[i].split())[1][-2] != '.':
            found_error = True
            print ("Always need one number after decimal point in energy!  ", data[i])
        if not (data[i][0].isupper()):
            found_error=True
    if ((found_error) or (found_end)):
        break
if (found_error==True):
    print ("Error in input file!   ", input_file,'    ', data[i],label_isospin)
    if (not found_end):
        print("No end found in input file ", input_file)
    sys.exit(1)
no_lines=end



# open final file
with open(results_name, 'w') as header:
    header.write('Element     Shape      A      Z    N  iso  Energy   Betas       Chi    Old_chi      '
                 'Adj Variables       Org Variables    magnitudes - final - orig\n')
    header.write('------------------------------------------------------------------------------------------'
                 '------------------------------------------------------------\n')
#Calculating number of max threads to be used

thread=0
cores=(multiprocessing.cpu_count()-1)
if no_lines < cores:
    thread=no_lines
else:
    thread=(multiprocessing.cpu_count()-1)



#saving opm script command to a array to be called with different threads
for i in range(0,end):
    temp_data.append("./opm.sh "+data[i]+" "+str(i+1)+" " + data2[i].lstrip()[0]+" "+version)
    
#bash command execution
def multiprocessing_func(x):
    string = temp_data[x]
    rc = call(string, shell=True)

if __name__ == '__main__':
    starttime = time.time()
    #Using semaphores to lock and unlock thread time slices, mainly to prevent concurrency issues like race condition as a lot of calculations occur in ecis loop
    pool = multiprocessing.Semaphore(thread)
    #Creating a pool of workers(threads)
    pool = multiprocessing.Pool(thread)
    #executing the saved array with bash command in terminal
    pool.map(multiprocessing_func, range(0,end))
    pool.close()
    #Closing and joining threads to prevent overflow
    pool.join
#    os.system('python3 emailer.py cred.inp')
#    os.system('python3 Email.py weppnesp@eckerd.edu')a

# Now sum up total chi^2
header.close()
total_chi=0.0
old_chi = 0.0
# open final file
with open(results_name, 'r') as header:
    data_sum= header.read().splitlines()
index = 0    
for lines in data_sum:
    index=index+1
    if index>2:
         chi=lines.split()
         total_chi=total_chi+float(chi[8])
         old_chi += float(chi[9])
header.close()
with open(results_name, 'a') as header:
    info = ('                                                Total Chi^2: ',total_chi, '   ',old_chi)
    header.write('\n')
    header.write("%4s  %10.2f %5s  %10.2f   \n" %info)
header.close()
for i in range(0,no_lines):
    st = 'rm -f '+remove(data[i])+'*'
    st2 = 'rm -f '+remove(data[i]+'State*.dat')
    # These lines below can be dangerous if data[i] is corrupted.
    # So much error checking above to make sure data[i] is ok
    os.system(st+"template.inp") # remove any last template files, data files or potential files
    os.system(st+".txt")
    os.system(st2)
print('That took {} seconds'.format(time.time() - starttime))
