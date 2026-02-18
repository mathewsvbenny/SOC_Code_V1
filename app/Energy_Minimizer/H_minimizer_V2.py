import numpy as np
import os,sys

from pathlib import Path
root_dir=Path(__file__).parent.parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from app.Energy_Minimizer.gen_H_TB import generate_H_TB_k_dep,TBHamiltonian
from app.SOC.create_H_SOC import generate_H_SOC 
# to get SOC + local magnetic field
from app.Basis_reordering.Transfer_Matrix import Trasfer_Matrix_spinful
 # helper function to transorm natural basis for SOC to wannier (orbital-major) orgering
from app.Unit_cell_composition.read_params import read_params_wrapper
#helper to read model parameters from a file
from app.Unit_cell_composition.read_k_space import read_k_space

class MagneticGroup:
    pass


class EnergyMinimizerParams:
    '''
    Docstring for EnergyMinimizerParams
    this class will hold all the parameters needed for energy minimizer to run
    1) win-file -> detailed information about the parameters of wannierization
    2) params-file-> initial values of the model  H_SOC hamitlonian
    3) magnetic_group-> an abstraction enfocing the symmetry of the magnetic order
    4) k-vector-> for the BZ integration
    5) TB_params-> holds a class TBHamiltonian (defined in gen_H_TB) containing all informations needed for the tb hamiltonian to be 
        constructed. That involves:
        1) composition-> to have number of wannier functions
        2) hoppings-> read from files hoppings intra- and inter- unit cell obtained from wannierization

    '''
    def __init__(self,win_file:str=None,param_file:str='params',magnetic_group:MagneticGroup=None,hr_files_list=[]):
        self.win_file=win_file
       
        if param_file=='params':
            print('Using the default param file "./params')

        if not os.path.exists(param_file):
            raise FileNotFoundError('Missing parameter file')
        else:
            print(f'{param_file} exists')
        self.param_file=param_file
        self.magnetic_group=magnetic_group
        self.k_space=np.zeros(3)
        self.TB_params=TBHamiltonian(self.win_file,*hr_files_list) #this holds the full information on TB model-> takes long to read so do it once


def Energy_minimizer(params:EnergyMinimizerParams,fixed_k:False)->np.ndarray:

    initial_param=read_params_wrapper(param_file=params.param_file, wannier_in_file=params.win_file) # get parameters to H_SOC
    # Set up- ferro magnetic order
    theta,phi=0,0
    for entries in initial_param['magnetic-field']:
        entries[-1] = phi
        entries[-2] = theta
        entries[-3] = 1.

    for entries in initial_param['SOC']:
        entries[-1]=0

    H_SOC= generate_H_SOC([params.win_file],initial_param)   # generate H_SOC (with optional local magnetic field)
    T_mat=Trasfer_Matrix_spinful([params.win_file])   # generate transfer matrix
    H_SOC_2=T_mat@H_SOC@T_mat.T              # transfer H_SOC to proper basis (orbital-major)
    
    #### Here should be the logic for minimization ####
    if fixed_k:
        k_vec_list=[np.zeros(3)]
    else:
        k_vec_list=read_k_space(params.win_file)
    
    energies=[]
    N_states=H_SOC_2.shape[0]
    filling=0.666
    max_state=int(N_states*filling)

    
    for _ in range(5):

        for k_it,k_vec in enumerate(k_vec_list):
            print(f'doing {k_it}/{len(k_vec_list)}')
            H=generate_H_TB_k_dep(params.TB_params,k_vec)+H_SOC_2
            energies.append([
                            theta,
                            phi,
                            np.sum(np.linalg.eigvalsh(H)[:1])#max_state])/max_state # this is integration up to fermi level
                            ])
        #increase theta
        theta += np.pi/4
        # pass the new value of theta to H_SOC parameters
        for entries in initial_param['magnetic-field']:
            entries[-2] = theta # the same for all atoms

        ####################################################
        # Recalulating of H_SOC with new theta value
        H_SOC= generate_H_SOC([params.win_file],initial_param)   # generate H_SOC (with optional local magnetic field)
        T_mat=Trasfer_Matrix_spinful([params.win_file])   # generate transfer matrix
        H_SOC_2=T_mat@H_SOC@T_mat.T              # transfer H_SOC to proper basis (orbital-major)
        

    return energies

#before rebasing






if __name__=="__main__":
    win_file='tests/test_cases/wannier90.win'
    hr_file_name='tests/test_cases/wannier90_up_hr.dat'
    hr_file_name2='tests/test_cases/wannier90_down_hr.dat' 
    param_name='tests/test_cases/params'
    initial_param=read_params_wrapper(param_file=param_name, wannier_in_file=win_file)
    print(type(initial_param))
    print(list(initial_param.keys()))
    print(type(initial_param['magnetic-field']))
    set_of_atoms=set()

    print('Before changes')
    for entries in initial_param['magnetic-field']:
        print(entries)
        set_of_atoms.add(entries[0])
    
    print(set_of_atoms)
    for entries in initial_param['magnetic-field']:
        if entries[0] == 'As':
            entries[-3]=0.4
        else:
            entries[-3]=0.1

    print('After chnages')
    for entries in initial_param['magnetic-field']:
        print(entries)
    
    ######################################################################
    print('Few steps of variation')
    params=EnergyMinimizerParams(win_file,param_name,None,[hr_file_name])
    params.k_space=np.zeros(3)
    res=Energy_minimizer(params,True)
    for energies in res:
        print(energies)
    
