# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:59:48 2024

@author: matth
"""

import hamiltonians_14_15 as ham15
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np

def get_transition_freqs_old(system, b_field):
    
    # Not sure about the B-field here --> components to magnitude? For now, just extracting the first component -- complied with hamiltonians_14_15
    hamiltonian = system.zfs_hamiltonian() + system.nitrogen_electric_quad_hamiltonian() + system.b_field_hamiltonian(b_field)# Ignoring hyperfine here as we only want the centres
    # print("\n",hamiltonian,"\n")
    
    psi_plus = qt.tensor(qt.basis(3,0), qt.basis(3,1))
    
    plus_energy = hamiltonian.matrix_element(psi_plus, psi_plus).real
    print(plus_energy)
    
    psi_ground = qt.tensor(qt.basis(3,1), qt.basis(system.multiplicity, 0)) # Only considering one nuclear spin state for now
    gs_energy = hamiltonian.matrix_element(psi_ground, psi_ground).real
    print(gs_energy)
    
    transition = plus_energy - gs_energy
    
    return transition
    
def get_transition_freqs_N14(system, b_field):
    hamiltonian = system.zfs_hamiltonian() + system.nitrogen_electric_quad_hamiltonian() + system.b_field_hamiltonian(b_field) # Ignoring hyperfine here as we only want the centres
    psi_plus = qt.tensor(qt.basis(3,0), qt.basis(3,0))
    psi_gs = qt.tensor(qt.basis(3, 1), qt.basis(3,0))
    plus_energy = hamiltonian.matrix_element(psi_plus, psi_plus)
    gs_energy = hamiltonian.matrix_element(psi_gs, psi_gs)
    e_trans = plus_energy - gs_energy
    
    psi_minus = qt.tensor(qt.basis(3,2), qt.basis(3,0))
    minus_energy = hamiltonian.matrix_element(psi_minus, psi_minus)
    e_trans_minus = minus_energy - gs_energy
    
    return b_field[2], e_trans, e_trans_minus

if __name__ == "__main__":
    nv_test = ham15.NvCentre(14)
    b_field_range = np.linspace(0, 10, 10000, endpoint=True)
    y_dat = []
    y_dat_minus = []
    x_dat = []
    for i in b_field_range:
        b_field = [0, 0, i]
        data_point_b = get_transition_freqs_N14(nv_test, b_field)[0]
        data_point_trans = get_transition_freqs_N14(nv_test, b_field)[1]
        x_dat.append(data_point_b)
        y_dat.append(data_point_trans)
        y_dat_minus.append(get_transition_freqs_N14(nv_test, b_field)[2])
    plt.plot(x_dat, y_dat)
    plt.plot(x_dat, y_dat_minus)
        
    plt.show()
