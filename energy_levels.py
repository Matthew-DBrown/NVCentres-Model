# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:26:12 2024

@author: matth
"""
import hamiltonians as ham
import hamiltonians_14_15 as ham15
import qutip as qt
import utility
import numpy as np
import matplotlib.pyplot as plt

def show_eigenenergies(hamiltonian, lines=False, pretty=False):
    eigenvalues = hamiltonian.eigenenergies()
    for index, val in enumerate(eigenvalues):
        print(f"Eigenenergy {index}: {val}")
    
    plt.figure(figsize=(9,9))
    plt.plot(np.linspace(0, len(eigenvalues), len(eigenvalues), endpoint=False), eigenvalues, 'b.', markersize=20)
    
    plt.xlabel("Eigenenergy index", fontsize=14)
    plt.ylabel("Frequency (Hz)", fontsize=14)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.grid(True, color='k', linestyle='--', linewidth=0.3, alpha=0.4)
    plt.legend()
    if lines:
        for y in eigenvalues:
            plt.hlines(y, 0, len(eigenvalues), linewidth=0.8, color='r')
    
    plt.show()
    

if __name__ == "__main__":
    nv = ham15.NvCentre(14)
    total_h = nv.total_hamiltonian([0,0,0])
    show_eigenenergies(total_h, lines=True, pretty=False)