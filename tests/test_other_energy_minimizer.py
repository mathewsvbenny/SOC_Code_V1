import sys, os
import numpy as np
import pytest
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from app.Energy_Minimizer.H_minimizer import Energy_minimizer, EnergyMinimizerParams, MagneticGroup,Energy_minimizer_gen_H_TB_k,Energy_minimizer_new_H_SOC
from app.Energy_Minimizer.gen_H_full import generate_H_Full,generate_H_TB_k_dep
from app.Energy_Minimizer.gen_H_TB import TBHamiltonian
from app.Unit_cell_composition.read_params import read_params_wrapper
from app.Unit_cell_composition.read_k_space import read_k_space
from app.Basis_reordering.Transfer_Matrix import Trasfer_Matrix_spinful
from app.SOC.create_H_SOC import generate_H_SOC
import numpy.typing as nty


def Energy_minimizer_test(params:EnergyMinimizerParams,fixed_k:False)->np.ndarray:
    initial_param=read_params_wrapper(param_file=params.param_file, wannier_in_file=params.win_file) # get parameters to H_SOC
    # Set up- ferro magnetic order
    theta,phi=0.,0.
    for entries in initial_param['magnetic-field']:
        entries[-1] = phi
        entries[-2] = theta
        entries[-3] = 100.
    for entries in initial_param['SOC']:
        entries[-1]=0.

    H_SOC= generate_H_SOC([params.win_file],initial_param) # generate H_SOC (with optional local magnetic field)
    T_mat=Trasfer_Matrix_spinful([params.win_file]) # generate transfer matrix
    H_SOC_2=T_mat@H_SOC@T_mat.T # transfer H_SOC to proper basis (orbital-major)

    energies=[]  
    theta=0.
    T_mat=Trasfer_Matrix_spinful([params.win_file]) # generate transfer matrix -> kand theta independent !
    H_ref=Energy_minimizer_gen_H_TB_k(params,[np.zeros(3)])[0]
    H_base=generate_H_TB_k_dep(params.TB_params,np.zeros(3))

    print("comparing TB-Hamiltonians")
    for base,ref in zip(H_base,H_ref):
        if np.any(np.abs(base-ref)):
            raise ValueError("TB-hamitlonians differ")
    
    
    for _ in range(5):
        for entries in initial_param['magnetic-field']:
            entries[-2] = theta # the same for all atoms
            #print(entries[-3:])

        H_SOC= generate_H_SOC([params.win_file],initial_param) # generate H_SOC (with optional local magnetic field)
        H_SOC_2=T_mat@H_SOC@T_mat.T # transfer H_SOC to proper basis (orbital-major)

        H_SOC_ref=Energy_minimizer_new_H_SOC(params,initial_param)

        for base,ref in zip(H_SOC_2,H_SOC_ref):
            if np.any(np.abs(base-ref)):
                raise ValueError("TB-hamitlonians differ")
    

        H= H_ref + H_SOC_ref


        energies.append([
            theta,
            phi,
            *np.linalg.eigvalsh(H)[:10]
        ])

        #increase theta
        theta += np.pi/4       

    return energies

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
            entries[-3]=100.
        else:
            entries[-3]=10.
    print('After chnages')
    for entries in initial_param['magnetic-field']:
        print(entries)

    ######################################################################
    
    print('Few steps of variation')
    params=EnergyMinimizerParams(win_file,param_name,None,[hr_file_name])
    res=Energy_minimizer_test(params,True)
    for energies in res:
        print(energies)
