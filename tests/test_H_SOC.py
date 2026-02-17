import numpy as np
import sys,os,pytest
from termcolor import colored
import time
from pathlib import Path
root_dir=Path(__file__).parent.parent
sys.path.append(str(root_dir))



from app.Misc.timing import timing
from app.Angular_momentum.angular_momentum import AngularMomentum
from app.Unit_cell_composition.UnitCell import get_L_from_orbitals_set_name
from app.Unit_cell_composition.read_win import composition_wrapper
from app.Unit_cell_composition.read_params import read_params, immerse_params_in_composition
from app.SOC.create_H_SOC import generate_H_SOC

@timing
def timed_generate_H_SOC(*filenames):
    res=generate_H_SOC(*filenames)
    return res

from app.Trash.create_H_SOC_V2 import generate_H_SOC_V2
@timing
def timed_generate_H_SOC_V2(*filenames):
    res=generate_H_SOC_V2(*filenames)
    return res

def check_difference(mat, mat2, size, ref):
    mat[np.absolute(mat)<1e-6] = 0
    mat2[np.absolute(mat2)<1e-6] = 0
    if ~np.any(mat - mat2):
        print("Test : ", colored("Passed", 'green'))
    else:
        exit("Test : Failed")


def test_print_orbital(subspace):
    first_letter=subspace[0][0]
    if (first_letter == 's'):
        print(colored("S orbital", 'red'))
    elif (first_letter == 'p'):
        print(colored("P orbital", 'blue'))
    elif (first_letter == 'd'):
        print(colored("D orbital", 'yellow'))
    else:
        raise Exception("Orbital unavailable!")


def calculate_H_SOC_ref(subspace, S_pauli):
    L_set = AngularMomentum(get_L_from_orbitals_set_name(subspace))
    L_set.to_Cartesian(subspace)
    H_SOC_ref = np.kron(L_set.x(),S_pauli.x()) + np.kron(L_set.y(),S_pauli.y()) + np.kron(L_set.z(),S_pauli.z())
    return H_SOC_ref
''''
if __name__=="__main__":

    test_case_loc='test_cases/'
    filename = test_case_loc+"wannier90_2_atoms.win"

    param_file = test_case_loc+"params_2_atoms"
    res=read_params(param_file)
    comp=composition_wrapper(filename)
    res2=immerse_params_in_composition(res,comp)

    H_SOC = generate_H_SOC([filename], params=res2)
    print("shape(H_SOC) = ", np.shape(H_SOC))
    #H_SOC[H_SOC < 1e-3] = 0
    np.set_printoptions(suppress=True)
    print("H_SOC = \n", H_SOC)
    
    print("\nH_SOC(Upper-left) = \n", np.diag(H_SOC[:6,:6]))
    print("\nH_SOC(Lower-right) = \n", np.diag(H_SOC[6:,6:]))
    
    exit()
    print(colored("spin up", 'red'), "=\n", H_SOC[::2, ::2])
    print(colored("spin down", 'red'), "=\n", H_SOC[1::2, 1::2])

    print(colored("SOC up/down", 'red'), "=\n", H_SOC[::2, 1::2])
    print(colored("SOC down/up", 'red'), "=\n", H_SOC[1::2, ::2])
'''

from app.Basis_reordering.Transfer_Matrix import Trasfer_Matrix_spinful
 # helper function to transorm natural basis for SOC to wannier (orbital-major) orgering
from app.Unit_cell_composition.read_params import read_params_wrapper
#helper to read model parameters from a file

class TestHSOCPhysics:
    param_file=os.path.join(root_dir,'tests','test_cases','params')
    win_file=os.path.join(root_dir,'tests','test_cases','wannier90.win')
    hr_file_1=os.path.join(root_dir,'tests','test_cases','wannier90_up_hr.dat')
    hr_file_2=os.path.join(root_dir,'tests','test_cases','wannier90_down_hr.dat')
    def test_magnetization_flip_ferro(self):
        initial_param=read_params_wrapper(param_file=self.param_file, wannier_in_file=self.win_file) # get parameters to H_SOC
        m_field=100
        for m_loc in initial_param['magnetic-field']:
            m_loc[-3]=m_field # |m|- module of mag-field
            m_loc[-2]=0 # theta
            m_loc[-1]=0 # phi
        H_SOC= generate_H_SOC([self.win_file],initial_param)   # generate H_SOC (with optional local magnetic field)
        diag_temp=np.diag(H_SOC)
        #Testing if diagonal is taggered
        for diag_terms in np.arange(len(diag_temp),step=2):
            assert diag_temp[diag_terms] == - diag_temp[diag_terms+1]
            assert np.abs( np.abs(diag_temp[diag_terms])-0.5*m_field)< 1e-6
        #flip the magnetic field
        for m_loc in initial_param['magnetic-field']:
            m_loc[-2]=np.pi # theta

        H_SOC= generate_H_SOC([self.win_file],initial_param)   # generate H_SOC (with optional local magnetic field)
        diag_temp_2=np.diag(H_SOC)
        # check if the diangonal elements from the two spin orientation add up to zero
        diff=[np.abs(el1+el2) for el1,el2 in zip(diag_temp,diag_temp_2)] 
        assert not np.any(diff)

    def test_obital_major_tran_of_H_SOC(self):
        initial_param=read_params_wrapper(param_file=self.param_file, wannier_in_file=self.win_file) # get parameters to H_SOC
        m_field=100
        orbs=list(set([m_loc[0] for m_loc in initial_param['magnetic-field']]))# extract different orb's
        print(f'These are the different atoms: {orbs}')
        for m_loc in initial_param['magnetic-field']:
            m_loc[-3]=m_field if m_loc[0]==orbs[0] else 0.2*m_field# |m|- module of mag-field
            m_loc[-2]=0.25*np.pi # theta
            m_loc[-1]=0.1 # phi
        H_SOC= generate_H_SOC([self.win_file],initial_param)  
        T_mat=Trasfer_Matrix_spinful([self.win_file])   # generate transfer matrix
        H_SOC_2=T_mat@H_SOC@T_mat.T              # transfer H_SOC to proper basis (orbital-major)
        
if __name__=="__main__":
    test_cases=TestHSOCPhysics()
    test_cases.test_obital_major_tran_of_H_SOC()
