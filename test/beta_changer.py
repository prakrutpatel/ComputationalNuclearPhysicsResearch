from decimal import *
import re
import numpy as np
# BOOLEANS FOR BETA CHANGER
# bool_beta_changer = True ---> set in potential program
# new_beta_changer= True  --> set in potential program
# The old beta changer (first one) made only one beta (the first) for each distortion (2+,3-,4+)
# It also changed the ECIS file to recognize this. Every 2+ level in C12, for example, had the same
# beta value. This was duplicated in ECIS. For calculating r_delta the other numbers
# were removed

# The new beta changer keeps all the beta values found. Keeps templete the same, Keeps ECIS the same.
# It is simplier, it only changes how r_delta is calculated. It takes a weighted average value instead
# For example it has a weighted average for both the 2+ levels

#For New Beta Change
# to Work Both Have to be true

#In case using beta changer using a newer class decimal to correctly format the betas
def formatFloat(val):
    getcontext().rounding= ROUND_DOWN
    getcontext().prec=2
    ret = Decimal(val) *Decimal(1.0)
    return ret

def unique(list1):  # find unique values in a list, maily used in beta changer
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def beta_changer(in_temp, spin, offset_rot, new_beta_changer, case, BOOL_HOLDER,data, E, beta):
    col1 =[]; col2 = []
    mod_beta = 1
    if spin == 'Rotational':
        mod_beta = 2
    beta_changer.new_beta = 0
    if case == 'on':
        print('Running potential program with beta changer using file ', in_temp)
        if new_beta_changer:
            print('Still using all Beta values, taking weighted average to figure out r_delta')
    with open(in_temp) as d:
        sleeping = d.read().splitlines()
    first_line = sleeping[0].split()
    temp_beta = int(first_line[0]) - 1
    # figuring no of betas to figure out how many lines i need to read and which lines contain beta information
    energy_level = [0.0] * temp_beta
    for z in range(0, (temp_beta * 2) - 1):
    # saving all energy states and beta values to an array so I can compare them to find unique betas and energy states
        if z % 2 == 0:
            offset = z
            offset2 = 0
            if spin == 'Rotational':
                offset = -temp_beta
                offset2 = -1
            mind = sleeping[3 + offset2 + z // mod_beta].split()
            energy_level[z // 2] = mind[-1]  # Save energy for averaging later
            # make sure correct level
            blown = sleeping[3 + (temp_beta * 2) + offset + offset_rot].split()
            col1.append(str(mind[0]) + str(mind[1]))
            if spin == 'Rotational':
                col2.append(blown[z // mod_beta])
            else:
                col2.append(blown[2])
    temp_col1 = unique(col1)  # find unique energy states
    for r in range(0, len(temp_col1)):  # find unique betas based on unique energy states and original values
        temp_index = []
        tempo_array = []
        temp_dex = []
        constr = temp_col1[r]
        for t in range(0, temp_beta):
            anotherstr = col1[t]
            if constr == anotherstr:
                temp_index.append(t)
                temp_dex.append(col2[t])
        lowest = temp_dex[0]
        for i in range(1, len(temp_index)):  # This statement now combines betas
            replace_statement = temp_dex[i] + ',' + lowest
            temp_dex[i] = replace_statement
            col2[temp_index[i]] = replace_statement
    for e in range(0, len(
            col2) * 2):  # col1 and col2 contain energy states and new beta information ex. .340,.250 where the second beta values is to replace the first one
        if e % 2 == 0:
            offset = e
            if spin == 'Rotational':
                offset = -temp_beta
            justforfun = int(e / 2)
            justforcol = int(3 + (temp_beta * 2) + offset + offset_rot)
            if col2[justforfun].find(',') != -1:  # check if a switch needs to occur
                beta_changer.new_beta = 1
                BOOL_HOLDER[justforfun] = "False"
                split_col = col2[justforfun].split(',')
                replace_word = split_col[0]
                temp_num = replace_word
                k = formatFloat(temp_num)
                replace_with = split_col[1]
                m = re.sub(replace_word, replace_with, sleeping[justforcol])  # beta replacement occurs here
                split_string = sleeping[justforcol].split()
                for i in range(0, len(split_string)):
                    dec_split_string = Decimal(split_string[i]) * Decimal('1.0')
                    if k == dec_split_string:
                        tempo_array.append(split_string[i])
                if not new_beta_changer:  # Don't change template file
                    for elements in tempo_array:
                        r = re.sub(elements, replace_with, sleeping[justforcol + 1])
                        l = re.sub(elements, replace_with, sleeping[justforcol + 1])
                        new_r_split = r.split()
                        l = format("{:1}{:9}{:10}{:10}{:10}{:10}{:10}{:10}".format(' ', new_r_split[0], new_r_split[1],
                                                                                   new_r_split[2], new_r_split[3],
                                                                                   new_r_split[4], new_r_split[5],
                                                                                   new_r_split[6]))
                    sleeping[justforcol] = m  # replacing old line with new ones
                    if spin == 'Vibrational':
                        sleeping[justforcol + 1] = l
    if not new_beta_changer:  # Don't Change array if new Beta Changer is on
        for o in range(0, len(data)):
            data[o] = sleeping[o]  # replace old template file with new one information
    if beta_changer.new_beta == 1:  # saving which beta values have been changed to create a new beta array
        no_true = 0
        energy_count = [[] for i in range(len(temp_col1))]
        for i in range(0, len(BOOL_HOLDER)):
            energy_count[temp_col1.index(col1[i])].append(i)
            if BOOL_HOLDER[i] == "True":
                no_true += 1
        beta_changer.new_beta_array = np.zeros(no_true)
        ghost = 0
        for o in range(0, len(BOOL_HOLDER)):
            if BOOL_HOLDER[o] == "True":
                if not new_beta_changer:
                    beta_changer.new_beta_array[ghost] = beta[o]
                else:
                    sum = 0.0
                    total_weight = 0.0
                    # Weighted average
                    for i in range(len(energy_count[ghost])):
                        new_beta = float(col2[energy_count[ghost][i]].partition(',')[0])
                        weight = 1.0 / (E + float(energy_level[energy_count[ghost][i]])) ** 2
                        sum += new_beta * weight
                        total_weight += weight
                    beta_changer.new_beta_array[ghost] = sum / total_weight
                ghost += 1
