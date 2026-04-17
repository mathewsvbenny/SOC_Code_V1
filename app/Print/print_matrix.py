import numpy as np
from datetime import datetime
import csv

def input_file_preamble_check(input_filename=[])->bool:
    '''
    Routnie to check if in case of two HR files
    the preambules match
    '''
    print(" len of input: ", len(input_filename))
    print(input_filename)
    if len(input_filename)==0:
        exit("Preambule missing")
    elif len(input_filename) ==1:
        return True
    elif len(input_filename)==2:
        file1, file2 = input_filename
        print("Comparing ", file1, " with ", file2)
        all_match = True
        lines_to_compare= np.ceil(int(np.loadtxt(input_filename[0], skiprows=2, max_rows=1)) /15)
        with open(file1) as f1, open(file2) as f2:
            for row_num, (line1, line2) in enumerate(zip(f1, f2)):
                if(row_num< 1): #skip the descrition line
                    continue
                if row_num >lines_to_compare:
                    break
                for v1,v2 in zip(line1.split(),line2.split()):
                    if v1 != v2:
                        print(f"Row {row_num} mismatch:")
                        all_match=False
        return all_match
    else:
        return False

def save_to_file(merged_hamiltonian, input_filename=[], output_filename: str ="output.dat"):
    '''
    Function saving merged hamiltonian to file and adding parameters 
    on the beginning of file from input_filename
    '''
        
    if input_file_preamble_check(input_filename):
        print( "Preabule check : PASSED")
    else:
        exit("Problem with the preabules of the two files")
    

    header_lines = []
    now = datetime.now()
    header_lines.append(f"Created on {now.strftime('%d%b%Y')} at {now.strftime('%H:%M:%S')}")

    spin_degeneracy = 2
    num_wann_og = int(np.loadtxt(input_filename[0], skiprows=1, max_rows=1))
    num_wann_new = num_wann_og * spin_degeneracy
    print("Doubling of the num_wann: ", num_wann_og, " -> ", num_wann_new)
    header_lines.append(int(num_wann_new))
    nrpts = np.loadtxt(input_filename[0], skiprows=2, max_rows=1)
    header_lines.append(int(nrpts))

    with open(output_filename, "w") as f:
        f.write(header_lines[0] + "\n")             #date
        f.write(str(int(header_lines[1])) + "\n")   #num_wann
        f.write(str(int(header_lines[2])) + "\n")   #nrpts

        nrpt_header_lines = np.loadtxt(input_filename[0], skiprows=3, max_rows=int(np.ceil(nrpts/15))-1)
        # "-1" above is to avoid reading the line witn number of entries different from 15
        for n_h_line in nrpt_header_lines:
            line = " ".join(str(int(x)) for x in n_h_line)
            f.write(line + "\n")

        nrpts_last_line = np.loadtxt(input_filename[0], skiprows=3+int(np.ceil(nrpts/15))-1, max_rows=1)
        # Sanity check to ensure that nrpts_last_line is a 1D array, even if it contains only one element
        if np.ndim(nrpts_last_line) == 0:
            nrpts_last_line = [nrpts_last_line]
        
        f.write(" ".join(str(int(x)) for x in nrpts_last_line) + "\n")

        for sets in merged_hamiltonian:
            line = " ".join(map(str, sets.to_Wannier()))
            f.write(line + "\n")
