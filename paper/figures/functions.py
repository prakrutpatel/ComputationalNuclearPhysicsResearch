import numpy as np
from statistics import mean



def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
def find_trigger(filename, word):
    if 'last' in filename:
        with open(filename) as f:
            counter = 0
            page = f.readlines()
            for line in page:
                counter += 1
                if word in line:
                    if counter > 1500:
                        return counter
        
    else:
        with open(filename) as f:
            counter = 0
            page = f.readlines()
            for line in page:
                counter += 1
                if word in line:
                    return counter

def import_data(filename, word, array):
    line_num = find_trigger(filename, word)
    array.append(np.loadtxt(filename, skiprows=((line_num) - 1), max_rows=1, usecols=(5)))
    return array

def best_fit(x, y):
    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)*mean(x)) - mean(x*x)))
    b = mean(y) - m*mean(x)
    line = [(m*x)+b for x in x]
    return line