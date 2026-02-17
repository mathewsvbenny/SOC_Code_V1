import sys, os
import re
import numpy as np
from termcolor import colored


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Angular_momentum'))
from Angular_momentum.ladder_operator import angular_momentum_matrices
from Angular_momentum.angular_momentum import AngularMomentum

#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Unit_cell_composition'))
from Unit_cell_composition.UnitCell import get_L_from_orbitals_set_name
from Unit_cell_composition.read_win import composition_wrapper
from Unit_cell_composition.read_params import read_params, immerse_params_in_composition

def check_input(atom_comp, atom_param, check_type):
    if (atom_comp.name != atom_param[0]
        or any(atom_comp.position[p] != atom_param[p+1] for p in np.arange(3))): # atoms mismatch 
        exit("Atoms mismatch while constructing H_SOC")
    if (check_type == 'SOC' and (len(atom_comp.split_orbitals_by_L()) != (len(atom_param) - 4))):
        exit("Atoms mismatch while constructing H_SOC")

def add_SOC(H_SOC, params, comp)->None:
    '''
    Extends H_SOC matrix by local and orbital-selective 
    spin-orbit Hamiltonian defined in params
    '''
    ref_point = 0 # ref point for H_SOC matrix
    S_Pauli = AngularMomentum(0.5)
    spin_degeneracy = S_Pauli.x().shape[0]
    if ('SOC' in params):
        for atom_comp, atom_param in zip(comp.composition, params['SOC']):
            check_input(atom_comp, atom_param, 'SOC')

            split_orb=atom_comp.split_orbitals_by_L()

            for iterator ,l_subspace in enumerate(split_orb):
                H_SOC_strength=atom_param[4+iterator]
                l=get_L_from_orbitals_set_name(l_subspace)
                L_op_set = AngularMomentum(l)
                L_op_set.to_Cartesian(l_subspace)

                H_SOC_in_L_subspace=H_SOC_strength*(np.kron(L_op_set.x(),S_Pauli.x())+np.kron(L_op_set.y(),S_Pauli.y())+np.kron(L_op_set.z(),S_Pauli.z()))
                for i in np.arange(H_SOC_in_L_subspace.shape[0]):            
                    for j in np.arange(H_SOC_in_L_subspace.shape[1]):
                        H_SOC[ref_point+i,ref_point+j] += H_SOC_in_L_subspace[i,j]
                ref_point += spin_degeneracy*len(l_subspace)

def add_magnetic_field(H_SOC, params, comp)-> None:
    '''
    Extends H_SOC matrix by local and orbital-selective 
    magnetic-field defined in params
    '''
    ref_point = 0 # ref point for H_SOC matrix
    S_Pauli = AngularMomentum(0.5)
    spin_degeneracy = S_Pauli.x().shape[0]
    if ('magnetic-field' in params):
        for atom_comp, atom_param in zip(comp.composition, params['magnetic-field']):
            check_input(atom_comp, atom_param, 'magnetic-field')

            split_orb=atom_comp.split_orbitals_by_L()

            for iterator, l_subspace in enumerate(split_orb):
                r       = atom_param[4+3*iterator] # "3" because each magnetic field is determined by 3 coeff's
                theta   = atom_param[5+3*iterator]
                phi     = atom_param[6+3*iterator]
                mag_field_x = r * np.sin(theta) * np.cos(phi)
                mag_field_y = r * np.sin(theta) * np.sin(phi)
                mag_field_z = r * np.cos(theta)
                l=get_L_from_orbitals_set_name(l_subspace)
                L_op_set = AngularMomentum(l)
                L_op_set.to_Cartesian(l_subspace)
                #print(f'problematic part l={l}, {L_op_set.x()} {type(L_op_set.x())} {L_op_set.x().shape}')
                if l==0:
                    id_mat=np.eye(1)
                else:
                    id_mat = np.eye(L_op_set.x().shape[0])


                H_MAG_in_L_subspace=np.kron(id_mat,S_Pauli.x()*mag_field_x)+np.kron(id_mat,S_Pauli.y()*mag_field_y)+np.kron(id_mat,S_Pauli.z()*mag_field_z)
                for i in np.arange(H_MAG_in_L_subspace.shape[0]):            
                    for j in np.arange(H_MAG_in_L_subspace.shape[1]):
                        H_SOC[ref_point+i,ref_point+j] += H_MAG_in_L_subspace[i,j]
                ref_point += spin_degeneracy*len(l_subspace)

def generate_H_SOC(filenames=[], params={}, print_details=False)-> np.ndarray:
    '''
    Calculates H_SOC matrix including magnetic field defined in params
    '''
    if print_details:
        print("Number of win's = ", len(filenames))
    if(len(filenames) == 0):
        win_file='wanner90.win'
    elif (len(filenames) == 1):	#if we pas 1 file, both are the same
        win_file = filenames[0]
    else:						#in other cases we have error
       raise IndexError("Too many win-files passed")

    S_Pauli = AngularMomentum(0.5)
    spin_degeneracy = S_Pauli.x().shape[0]
    comp = {}
    comp = composition_wrapper(win_file)
    if print_details:
        comp.print_composition()

    num_wann=comp.get_num_wann()
    num_spin_wann=spin_degeneracy*num_wann

    H_SOC = np.zeros((num_spin_wann,num_spin_wann), dtype=complex) # Inititate proprely sized zero-matrix

    add_SOC(H_SOC, params, comp) # add SOC
    add_magnetic_field(H_SOC, params, comp) # add mag-field

    return H_SOC