import sys,os
import numpy as np
from pathlib import Path
import pytest
root_dir=Path(__file__).parent.parent
sys.path.append(str(root_dir))

from app.Basis_reordering.Transfer_Matrix import Trasfer_Matrix_spinful
from app.Unit_cell_composition.read_win import composition_wrapper


@pytest.mark.parametrize('file', ['mnte.win','wannier90.win', 'wannier90_V3.win'])
class TestTransferMatrix:
    test_root=os.path.join(root_dir,'tests','test_cases')


    def test_square(self,file):
        print("\nReading from :", file)
        l=Trasfer_Matrix_spinful([os.path.join(self.test_root,file)],print_details=False)
        sq= l.T@l
        assert not np.any(sq-np.eye(l.shape[0]))

    def test_illustrate_reordeing(self,file):
        comp = composition_wrapper(os.path.join(self.test_root,file))
        orb_list_print=[]
        orb_list=[]    
        for it,atom in enumerate(comp):
            for orb in atom.orbitals: #artificialy create spinful basis
                orb_list_print.append(f'{it}_{orb}' + "_up")
                orb_list_print.append(f'{it}_{orb}' + "_down")
            orb_list=orb_list+ atom.orbitals

        orb_list=np.array(orb_list)
        orb_list_print=np.array(orb_list_print)

        #create a new reordered list
        new_orb_list=[]
        l=Trasfer_Matrix_spinful([os.path.join(self.test_root,file)],print_details=False)
        reordered= l@np.arange(len(orb_list_print)) #vector after trasnformation
        for new_pos in reordered:
            new_orb_list.append(orb_list_print[int(new_pos)]) #change positions to str-values
        
        print(f'{"pre".center(15)} {"after".center(15)}')
        for ent1,ent2 in zip(orb_list_print,new_orb_list):
            print(f'{str(ent1).center(15)} {str(ent2).center(15)}')



if __name__=="__main__":
    test_lookup=TestTransferMatrix()
    test_lookup.test_illustrate_reordeing('wannier90.win')
        
