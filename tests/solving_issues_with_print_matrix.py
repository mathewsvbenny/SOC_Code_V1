if __name__ == "__main__":
    from datetime import datetime
    import numpy as np  
    #input_filename=['../tests/test_cases/wannier90_hr.dat' ]
    input_filename=['wannier90_hr.dat' ]
    spin_degeneracy = 2
    num_wann_og = int(np.loadtxt(input_filename[0], skiprows=1, max_rows=1))
    num_wann_new = num_wann_og * spin_degeneracy
    print("Doubling of the num_wann: ", num_wann_og, " -> ", num_wann_new)
    #header_lines.append(int(num_wann_new))
    nrpts = np.loadtxt(input_filename[0], skiprows=2, max_rows=1)
    print(f'Numerb of points {nrpts}')
    #header_lines.append(int(nrpts))
    #
    # 
    #  
    nrpt_header_lines = np.loadtxt(input_filename[0], skiprows=3, max_rows=int(np.ceil(nrpts/15))-1)
    nr_pts_read=0
    
    for x in nrpt_header_lines:
        nr_pts_read += len(x) 
    

    print(f'Number of header lines {nrpt_header_lines} ({nr_pts_read})')
    nrpts_last_line=np.loadtxt(input_filename[0], skiprows=3+int(np.ceil(nrpts/15))-1, max_rows=1)
    #print(type(nrpts_last_line))
    #try:
    #    print(f' Numer of entries in the last line {len(nrpts_last_line)}')
    if np.ndim(nrpts_last_line) == 0:
        nrpts_last_line = [nrpts_last_line]

    print(" ".join(str(int(x)) for x in nrpts_last_line) + "\n")

    
    exit()
    '''
    with open(output_filename, "w") as f:
        f.write(header_lines[0] + "\n")             #date
        f.write(str(int(header_lines[1])) + "\n")   #num_wann
        f.write(str(int(header_lines[2])) + "\n")   #nrpts

        nrpt_header_lines = np.loadtxt(input_filename[0], skiprows=3, max_rows=int(np.ceil(nrpts/15))-1)
        for n_h_line in nrpt_header_lines:
            line = " ".join(str(int(x)) for x in n_h_line)
            f.write(line + "\n")
        f.write(" ".join(str(int(x)) for x in nrpts_last_line) + "\n")

        for sets in merged_hamiltonian:
            line = " ".join(map(str, sets.to_Wannier()))
            f.write(line + "\n")
    '''