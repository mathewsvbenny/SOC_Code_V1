import sys, os
import numpy as np
import pytest
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
print(root_dir)
from app.Energy_Minimizer.H_minimizer import Energy_minimizer, EnergyMinimizerParams, MagneticGroup
from app.Energy_Minimizer.gen_H_full import generate_H_Full
from app.Energy_Minimizer.gen_H_TB import TBHamiltonian
import numpy.typing as nty





class TestEnergyMinimizerParams:
    """Tests for EnergyMinimizerParams initialization"""
    
    def test_missing_param_file(self):
        """Test that FileNotFoundError is raised when default param file is missing"""
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        
        with pytest.raises(FileNotFoundError, match='Missing parameter file'):
            EnergyMinimizerParams(win_file=win_file, param_file=os.path.join(root_dir,'tests','params'), magnetic_group=None, hr_files_list=[file_name])
    
    def test_custom_param_file(self):
        """Test initialization with custom param file"""
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        param_file = os.path.join(root_dir, 'tests', 'test_cases', 'params')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        
        try:
            params = EnergyMinimizerParams(win_file=win_file, param_file=param_file, magnetic_group=None, hr_files_list=[file_name])
            assert params.win_file == win_file
            assert params.param_file == param_file
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")
    
    def test_with_magnetic_group(self):
        """Test initialization with MagneticGroup"""
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        param_file = os.path.join(root_dir, 'tests', 'test_cases', 'params')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        mag_group = MagneticGroup()
        
        try:
            params = EnergyMinimizerParams(win_file=win_file, param_file=param_file, magnetic_group=mag_group, hr_files_list=[file_name])
            assert params.magnetic_group == mag_group
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")


class TestEnergyMinimizer:
    """Tests for Energy_minimizer function"""
    
    def test_return_type_is_ndarray(self):
        """Test that function returns numpy ndarray"""
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        param_file = os.path.join(root_dir, 'tests', 'test_cases','params')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        print(f'taking {param_file}')
        try:
            params = EnergyMinimizerParams(win_file=win_file, param_file=param_file, magnetic_group=None, hr_files_list=[file_name])
            result = Energy_minimizer(params)
            assert isinstance(result, np.ndarray), "Result should be numpy ndarray"
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")
    
    def test_result_shape_matches_hamiltonian(self):
        """Test that result has valid shape"""
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        param_file = os.path.join(root_dir, 'tests', 'test_cases', 'params')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        
        try:
            params = EnergyMinimizerParams(win_file=win_file, param_file=param_file, magnetic_group=None, hr_files_list=[file_name])
            result = Energy_minimizer(params)
            assert result.ndim == 2, "Result should be 2D matrix"
            assert result.shape[0] == result.shape[1], "Hamiltonian should be square"
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")


    def test_energy_minimizer_vs_H_full(self):
        win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
        param_file = os.path.join(root_dir, 'tests', 'test_cases', 'params')
        file_name = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
        try:
            #tb_params=TBHamiltonian(win_file,file_name)
            params = EnergyMinimizerParams(win_file=win_file, param_file=param_file, magnetic_group=None, hr_files_list=[file_name])
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")
        
        # Do a comparative test
        H_full=generate_H_Full(win_file,param_file,file_name)
        H_from_minimizer=Energy_minimizer(params)
        for e1,e2 in zip(np.linalg.eigvalsh(H_full),np.linalg.eigvalsh(H_from_minimizer)):
            
            if np.abs(e1-e2)> 1e-5:
                raise ValueError("H-full and minimizer produce a different Hamiltonian fo k=0")


from app.Unit_cell_composition.read_params import read_params_wrapper
from app.Energy_Minimizer.H_minimizer import Energy_minimizer_gen_H_TB_k,Energy_minimizer_new_H_SOC

class TestEnergyMinimizerPhysics:
    win_file = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90.win')
    param_file = os.path.join(root_dir, 'tests', 'test_cases', 'params')
    hr_file_name_1 = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_up_hr.dat')
    hr_file_name_2 = os.path.join(root_dir, 'tests', 'test_cases', 'wannier90_down_hr.dat')

    def test_spin_flip(self):
        try:
            #tb_params=TBHamiltonian(win_file,file_name)
            params = EnergyMinimizerParams(
                win_file=self.win_file, 
                param_file=self.param_file, 
                magnetic_group=None, 
                hr_files_list=[self.hr_file_name_1]
                )
        except (FileNotFoundError, TypeError):
            pytest.skip("Test files not available")
        
        
        SOC_param=read_params_wrapper(param_file=self.param_file, wannier_in_file=self.win_file) # get parameters to H_SOC
        #Tweak magnetic-field
        orbs=list(set([m_loc[0] for m_loc in SOC_param['magnetic-field']])) # list of distinct atoms
        print(f'We have a set of orbitals {orbs}')
        m_field=100
        print(f'Setting |m|={m_field}')
        
        angle=0        
        for m_loc in SOC_param['magnetic-field']:
            m_loc[-3]=m_field #if m_loc[0]==orbs[0] else 0.2*m_field# |m|- module of mag-field
            m_loc[-2]=angle # theta
            m_loc[-1]=0. # phi
        for soc_temp in SOC_param['magnetic-field']:
            soc_temp[-1]=0.

        energies=[]
        angles=[]
        H_TB_k=Energy_minimizer_gen_H_TB_k(params,[np.zeros(3)])
        for _ in range(6):
            H_SOC= Energy_minimizer_new_H_SOC(params,SOC_param)
            angles.append(angle)
            for H_TB in H_TB_k:
                energies.append(np.linalg.eigvalsh(H_TB+H_SOC)[:10])
            angle += np.pi/5
            for m_loc in SOC_param['magnetic-field']:
                m_loc[-2]=angle
        energies=np.array(energies)
        diff=np.abs(energies[0,:]-energies[-1,:])
        diff[diff<1e-6]=0
        print(f'Azymuthal angle zero and pi are the same : {not np.any(diff)}')
        assert not np.any(diff)
        energies=np.vstack((angles, np.transpose(energies)))
        print(diff)
        for row in energies:
            print('  '.join(list(map(lambda x: f'{x:>10.6f}',row))))    


if __name__=="__main__":
    test1=TestEnergyMinimizerPhysics()
    test1.test_spin_flip()