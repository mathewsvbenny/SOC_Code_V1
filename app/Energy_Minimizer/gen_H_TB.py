import numpy as np
import os,sys
app_loc=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(app_loc)
from Unit_cell_composition.create_Hamiltonian import create_hamiltonian,Wannier_data
from Unit_cell_composition.read_win import composition_wrapper 
from Unit_cell_composition.UnitCell import UnitCell


class TBHamiltonian:
    '''
    Docstring for TBHamiltonian
    A class holding all the informations needed to create the 
    Tightbinding hamilonina from wannierization:
    Memebers:
    1)comp-> composition of the unit-cell
    2)hoppings-> all the hoppings from wannierization read from the file(s)
    '''
    def __init__(self,win_file:str=None,*hr_files):
        merged=create_hamiltonian(*hr_files)
        if win_file is None:
            win_file='wanner90.win'
            print(f'Trying default win-file name ({win_default})')
        if not os.path.exists(win_file):
            raise FileNotFoundError(f'{win_file} was not found, check where you are')
        print(f'Reading compositon from {win_file}')
        self.comp=composition_wrapper(win_file)
        #print("Truncating zero-valued hopping elements")
        self.hoppings=merged

    def truncate(self, min_val: float=1e-3):
        print(f'Truncating hoppings < {min_val}')
        self.hoppings=list(filter(lambda x: np.abs(x.hop)>1e-3,self.hoppings))


def generate_H_TB(win_file:str=None,*hr_files):
    """
    Generate matrix with only the 
    wannnierized TB parametes
    
    Needed parameters:
    win_file: path to the wannier90 parameters
    hf_files: paths to spin_up and spin_down wannierization results

    If no parametrs are given then attemting to use the default names
    """

    H_TB_params=TBHamiltonian(win_file,*hr_files)
    
    num_wann=H_TB_params.comp.get_num_wann()
    spin_degeneracy=2
    num_spin_wann=spin_degeneracy*num_wann

    H_TB = np.zeros((num_spin_wann,num_spin_wann), dtype=complex) # Inititate proprely sized zero-matrix
    for sets in H_TB_params.hoppings:
          if [sets.x,sets.y,sets.z] == [0,0,0]: 
            ind_1=sets.o1-1 # to python convention
            ind_2=sets.o2-1 # to python convension
            H_TB[ind_1][ind_2] =sets.hop

    return H_TB



def generate_H_TB_k_dep(H_TB_params:TBHamiltonian,k_vec=np.zeros(3),min_val: float =None):
    '''
    Docstring for generate_H_TB_k_dep
    
    extenstion of the local TB hamiltonian (r=0,0,0) to account for all
    the hoppings, that is also the inter-unitcell, multiplied by the
    phase factor from the Fourier Transform
    '''
    num_wann=H_TB_params.comp.get_num_wann()
    spin_degeneracy=2
    num_spin_wann=spin_degeneracy*num_wann
    if not min_val is None:
        H_TB_params.truncate(min_val) 

    H_TB = np.zeros((num_spin_wann,num_spin_wann), dtype=complex) # Inititate proprely sized zero-matrix
    for sets in H_TB_params.hoppings:
        phase=np.exp(1.j*np.dot(k_vec,np.array([sets.x, sets.y,sets.z])))
        ind_1=sets.o1-1 # to python convention
        ind_2=sets.o2-1 # to python convension
        H_TB[ind_1][ind_2] += phase*sets.hop

    return H_TB



if __name__=="__main__":
    win_file='tests/test_cases/wannier90.win'
    file_name='tests/test_cases/wannier90_up_hr.dat'
    file_name2='tests/test_cases/wannier90_down_hr.dat'
    res=generate_H_TB(win_file,file_name,file_name)
    print(res.shape)
    res_e=np.linalg.eigvalsh(res)
    print(f'first 5 eigenvalues(local)= {res_e[:5]}')
    tb_params=TBHamiltonian(win_file,file_name,file_name)
    res_2=generate_H_TB_k_dep(tb_params,np.zeros(3))
    res_2_e=np.linalg.eigvalsh(res_2)
    print(f'first 5 eigenvalues(full)= {res_2_e[:5]}')

    
