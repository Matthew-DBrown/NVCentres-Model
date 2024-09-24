# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:25:01 2024

@author: matth
"""
import numpy as np
import qutip as qt
import scipy as sc
from scipy import constants

# Defining physical constants
mu_b = constants.value(constants.find("bohr")[0]) # Bohr magneton
e_gyro = constants.value(constants.find("gyro")[1])*10**6 # Electron gyromagnetic ration in Hz/Tesla
axial_hyp = {14:-2.14e6, 15:3.03e6} # Units Hz
trans_hyp = {14:-2.7e6, 15:3.65e6} # Units Hz


class NvCentre:
    
    def __init__(self, isotope):
        
        self.isotope = isotope
        if isotope not in [14, 15]:
            raise ValueError("The nitrogen isotope must be N-14 or N-15.")
        

        self.nuclear_spin = 1
        if isotope == 15:
            self.nuclear_spin = 0.5
            
        self.axial_hyp = axial_hyp[self.isotope]
        self.trans_hyp = trans_hyp[self.isotope]
        self.nuc_electric_axial_quad = -5.01e6 # Units Hz
        self.zfs = 2.878e9 # Units Hz
        self.multiplicity = 2*self.nuclear_spin + 1 # As in spectroscopic notation
        
    def zfs_hamiltonian(self):
        s_squared = qt.spin_Jx(1)**2 + qt.spin_Jy(1)**2 + qt.spin_Jz(1)**2
        h = self.zfs*(qt.spin_Jz(1)**2 - s_squared/3)
        return qt.tensor(h, qt.qeye(self.multiplicity))
    
    def nitrogen_electric_quad_hamiltonian(self):
        '''
        P[I_z**2 - (I**2)/3]
        '''
        if self.isotope == 15:
            h = 0 # This is needed otherwise the dimensions of the matrix is incompatible with the other hamiltonians
        
        else:        
            I_x = qt.spin_Jx(self.nuclear_spin)
            I_y = qt.spin_Jy(self.nuclear_spin)
            I_z = qt.spin_Jz(self.nuclear_spin)
            I_squared = I_x**2 + I_y**2 + I_z**2
            h = self.nuc_electric_axial_quad*(I_z**2 - I_squared/3)
            h = qt.tensor(h, qt.qeye(self.multiplicity))
            
        return h
    
    def nitrogen_hyperfine_hamiltonian(self):
        I_x = qt.spin_Jx(self.nuclear_spin)
        I_y = qt.spin_Jy(self.nuclear_spin)
        I_z = qt.spin_Jz(self.nuclear_spin)
        S_x, S_y, S_z = qt.spin_Jx(1), qt.spin_Jy(1), qt.spin_Jz(1)
        h_axial = self.axial_hyp*qt.tensor(S_z, I_z)
        h_transverse = self.trans_hyp*(qt.tensor(S_x, I_x) + qt.tensor(S_y, I_y))
        return h_axial + h_transverse
    
    def b_field_hamiltonian(self, B):
        '''
        Takes the magnetic field and returns the hamiltonian.
        Parameters
        ----------
        B : numpy array
            [B_x, B_y, B_z].

        Returns
        -------
        Hamiltonian

        '''
        B_x, B_y, B_z = B[0], B[1], B[2]
        S_x, S_y, S_z = qt.spin_Jx(1), qt.spin_Jy(1), qt.spin_Jz(1)
        h = e_gyro*(B_x*S_x + B_y*S_y + B_z*S_z)
        return qt.tensor(h, qt.qeye(self.multiplicity))
    
    def total_hamiltonian(self, b_field=None):
        h1 = self.zfs_hamiltonian()
        h2 = self.nitrogen_electric_quad_hamiltonian()
        h3 = self.nitrogen_hyperfine_hamiltonian()
        if b_field == None:
            b_field = [0,0,0]
        else:
            pass
        h4 = self.b_field_hamiltonian(b_field)
        return h1 + h2 + h3 + h4
        
        