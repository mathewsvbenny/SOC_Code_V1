import numpy as np
import os,sys
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
from app.Unit_cell_composition.read_win import composition_wrapper

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



from scipy.optimize import minimize
import random
def Energy_minimizer(params:EnergyMinimizerParams,fixed_k:False,m:float=0,l_SOC:float=0.)->np.ndarray:
    if fixed_k:
        k_vec_list=[np.zeros(3)]
    else:
        k_vec_list=read_k_space(params.win_file)
      
    H_TB_k_list=Energy_minimizer_gen_H_TB_k(params,k_vec_list)
    #filling-related stuff -> hihghest occupierd state 
    n_states=H_TB_k_list[0].shape[0]
    max_filled_state=int(np.ceil(n_states*params.filling))
    
    #### Here should be the logic for minimization ####
    H_SOC_param=read_params_wrapper(param_file=params.param_file, wannier_in_file=params.win_file) # get parameters to H_SOC
    for SOC_internals in H_SOC_param['SOC']:
        SOC_internals[-1]=l_SOC
    magnetic_moment=m
    ### Get all As atoms
    #unit_cell=composition_wrapper(file_name=params.win_file)
    #num_of_as=0
    #for at in unit_cell:
    #    if at.name=='As':
    #        num_of_as +=1
    #print(f'There is {num_of_as} As atoms')
    num_of_as=1

    def minimizer_internals(x):
        nonlocal H_TB_k_list,H_SOC_param,magnetic_moment
        as_iter=0
        for mag_field_internals in H_SOC_param['magnetic-field']:
            if mag_field_internals[0] == 'As':
                
                mag_field_internals[-3] = magnetic_moment
                mag_field_internals[-2] = x[as_iter*2+0]
                mag_field_internals[-1] = x[as_iter*2+1]
                #as_iter +=1
                #if as_iter == num_of_as:
                #    raise ValueError("Something is wrong with countng As")

            else:
                mag_field_internals[-3]=0.


        H_SOC=Energy_minimizer_new_H_SOC(params,H_SOC_param)
        energies=[]
        for H_mat in tqdm(H_TB_k_list,desc='Integrating the k-dependent H with SOC terms'):
            H=H_mat+H_SOC
            energies.append(np.sum(np.linalg.eigvalsh(H)[:max_filled_state]))
        
        number_of_dimensions=3
        res=[np.average(energies)/(number_of_dimensions*2.*np.pi)]
        print(f'Energy at ', end='')
        for it in range(0,len(x),2):
            print(f'theta={x[it]}, phi={x[it+1]} ',end='')
        print(f' is  {res[0]:.6f}')
        
        return res

    res=minimize(minimizer_internals,
                 [random.random() for _ in range(2*num_of_as)],
                 method='nelder-mead',
                 bounds=[b for _ in range(num_of_as) for b in [(0.,np.pi),(0.,2.*np.pi)]],
                 options={'xatol': 1e-8, 'disp': True,'maxiter': 300*2*num_of_as}
                )
    return res.x


if __name__=="__main__":
    win_file='tests/test_cases/wannier90.win'
    hr_file_name='tests/test_cases/wannier90_up_hr.dat'
    hr_file_name2='tests/test_cases/wannier90_down_hr.dat' 
    param_name='tests/test_cases/params'
    params=EnergyMinimizerParams(win_file,param_name,None,[hr_file_name])

    params.min_val=1e-2
    params.filling=0.66

    final_outcome=[]
    for l_SOC in np.arange(0.75, 1.1,0.05):
        res=Energy_minimizer(params,False,m=1.,l_SOC=l_SOC)
        final_outcome.append([f'{x:.6f}' for x in [l_SOC,*res]])
        print(final_outcome[-1])
    with open('SOC_in_FM.dat','a') as f:
        for res in final_outcome:
            f.write(' '.join(res)+'\n')