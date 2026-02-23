import numpy as np
import os,sys
import datetime
from tqdm import tqdm

from pathlib import Path
root_dir=Path(__file__).parent.parent.parent
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
    def __init__(self,win_file:str=None,param_file:str='params',magnetic_group:MagneticGroup=None,hr_files_list=[],min_val:float=None):
        self.win_file=win_file
       
        if param_file=='params':
            print('Using the default param file "./params')

        if not os.path.exists(param_file):
            raise FileNotFoundError('Missing parameter file')
        else:
            print(f'{param_file} exists')
        self.param_file=param_file
        self.magnetic_group=magnetic_group
        self.k_space=[np.zeros(3)]
        self.TB_params=TBHamiltonian(self.win_file,*hr_files_list) #this holds the full information on TB model-> takes long to read so do it once
        if not min_val is None:
            self.TB_params.truncate(min_val)
        self._min_val=min_val
        self._filling=1.


    @property
    def min_val(self):
        return self._min_val
    
    @min_val.setter
    def min_val(self,min_val:float):
        if isinstance(min_val,float):
            self._min_val=min_val
            self.TB_params.truncate(min_val)
        else:
            raise TypeError("Min_val should be a float")

    @property
    def filling(self)->None:
        return self._filling
    @filling.setter
    def filling(self,n:float|int)->None:
        '''
        Docstring for filling
        
        :param n: fraction of all states that is occupied
        :type n: float,int
        '''
        if not isinstance(n,(float,int)):
            raise ValueError("Filling should be in [0,1] range")
        else:
            self._filling=n


def Energy_minimizer_gen_H_TB_k(params:EnergyMinimizerParams,k_vec_list:list[np.ndarray])->list[np.ndarray]:
    '''
    Docstring for Energy_minimizer_gen_H_TB_k
    Returns the list of H_TB for each k-vector from the k_vec_list
    If memory can hold it it will speed-up things a lot more, ToBeSeen
    '''
    res= [generate_H_TB_k_dep(params.TB_params,k_vec) for k_vec in tqdm(k_vec_list,desc='Generating list of k-dependent Hamiltonians')]
    return res

def Energy_minimizer_new_H_SOC(params:EnergyMinimizerParams,SOC_param:dict[str,list[str|float]]=None):
    '''
    Generates a new SOC matrix in the orbital-major basis, based on the fact
    if user gave CUSTOM SOC params or not, then use the provided by in params-file
    '''
    if SOC_param is None:
        SOC_param=read_params_wrapper(param_file=params.param_file, wannier_in_file=params.win_file) # get parameters to H_SOC
    H_SOC= generate_H_SOC([params.win_file],SOC_param)   # generate H_SOC (with optional local magnetic field)
    T_mat=Trasfer_Matrix_spinful([params.win_file])   # generate transfer matrix
    H_SOC_2=T_mat@H_SOC@T_mat.T 
    return H_SOC_2


def Energy_minimizer(params:EnergyMinimizerParams,fixed_k:False)->np.ndarray:
    if fixed_k:
        k_vec_list=[np.zeros(3)]
    else:
        k_vec_list=read_k_space(params.win_file)
      
    H_TB_k_list=Energy_minimizer_gen_H_TB_k(params,k_vec_list)
    

    H_SOC_param=read_params_wrapper(param_file=params.param_file, wannier_in_file=params.win_file) # get parameters to H_SOC

    H_SOC=Energy_minimizer_new_H_SOC(params,H_SOC_param)
    #max_filling
    n_states=H_SOC.shape[0]
    max_filled_state=int(np.ceil(n_states*params.filling))

    #### Here should be the logic for minimization ####
    energies=[]
    for H_mat in tqdm(H_TB_k_list,desc='Integrating the k-dependent H with SOC terms'):
        H=H_mat+H_SOC
        energies.append(np.sum(np.linalg.eigvalsh(H)[:max_filled_state]))
    
    number_of_dimensions=3
    return [np.average(energies)/(number_of_dimensions*2.*np.pi)]




if __name__=="__main__":
    win_file='tests/test_cases/wannier90.win'
    hr_file_name='tests/test_cases/wannier90_up_hr.dat'
    hr_file_name2='tests/test_cases/wannier90_down_hr.dat' 
    param_name='tests/test_cases/params'
    params=EnergyMinimizerParams(win_file,param_name,None,[hr_file_name])
    params.min_val=1e-2
    res=Energy_minimizer(params,False)
    print(np.array(res))