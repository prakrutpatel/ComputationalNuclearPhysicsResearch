import multiprocessing
from multiprocessing import pool
import sys
from subprocess import call
import time
import os


# open final file
with open('results_final_fit.txt', 'w') as header:
    header.write('Element     Shape      A        Z      N      Energy   Betas       Chi    Old_chi        Adj Variables         Org Variables\n')
    header.write('----------------------------------------------------------------------------------------------------------------------------\n')

temp_data = []
no_thread = multiprocessing.cpu_count()
input_file = sys.argv[1]
version = sys.argv[2]
with open(input_file) as f:
    data = f.read().splitlines()
for i in range(0,len(data)):
    if 'end' in data[i]:
        end = i
        break
no_lines=(len(data)-1)
thread=0
cores=(multiprocessing.cpu_count()-1)
if no_lines < cores:
    thread=no_lines
else:
    thread=(multiprocessing.cpu_count()-1)



for i in range(0,i):
    temp_data.append("./opm_final.sh "+data[i]+" "+str(i+1)+" "+version)

def multiprocessing_func(x):
    string = temp_data[x]
    rc = call(string, shell=True)

if __name__ == '__main__':
    starttime = time.time()
    pool = multiprocessing.Semaphore(thread)
    pool = multiprocessing.Pool(thread)
    pool.map(multiprocessing_func, range(0,end))
    pool.close()
    pool.join
    print('That took {} seconds'.format(time.time() - starttime))
    # Now sum up total chi^2
    header.close()
    total_chi=0.0
    old_chi = 0.0
    # open final file
    with open('results_final_fit.txt', 'r') as header:
        data = header.read().splitlines()
    index = 0
    for lines in data:
        index=index+1
        if index>2:
             chi=lines.split()
             total_chi=total_chi+float(chi[7])
             old_chi += float(chi[8])
    header.close()
    with open('results_final_fit.txt', 'a') as header:
        info = ('                                                Total Chi^2: ',total_chi, old_chi)
        header.write('\n')
        header.write("%4s  %10.2f    %10.2f \n" %info)
    header.close()



