# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:26:12 2024

@author: matth
"""
import hamiltonians as ham
import hamiltonians_14_15 as ham15
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def lorentzian(x, x_0, gamma, amplitude):
    return amplitude*(1/np.pi)*(0.5*gamma/((np.array(x) - x_0)**2 + (0.5*gamma)**2))

def plot_odmr(system, h):
    eigenvalues = h.eigenenergies()
    def_linewidth = 1e6 # Defaul linewidth
    f_array = np.linspace(1.8e9, 3.5e9, 1000000, endpoint=True)
    amplitude = 0.05
    scale = 1e7 # Arbitrary fix for now??
    lor_centres = []
    for i in range(system.multiplicity):
        for j in [0,2]:
            psi_ground = qt.tensor(qt.basis(3,1), qt.basis(system.multiplicity, j))
            psi_electron = qt.tensor(qt.basis(3, j), qt.basis(system.multiplicity, i))
            gs_energy = h.matrix_element(psi_ground, psi_ground)
            electron_energy = h.matrix_element(psi_electron, psi_electron)
            peak = electron_energy - gs_energy
            lor_centres.append(peak)
    
    line_data = []
    for f0 in lor_centres:
        line_data.append(np.array(lorentzian(f_array, abs(f0), def_linewidth, amplitude)))
    print(lor_centres)
    line_data = np.vstack(line_data)
    plot_data = np.sum(line_data, axis=0)
    plt.plot(f_array, 1-plot_data*scale, 'm')
    plt.xlim(0.9*lor_centres[0].real, 1.1*lor_centres[3].real)
    plt.xlabel("Frequency (Hz)", fontsize=13)
    plt.ylabel("Intensity (Arbitrary Units)", fontsize=13)
    plt.title(f"Simulated ODMR Spectrum for a Single NV Centre: N-{system.isotope} / B-field (T): {system.b_field}", fontsize=13)
    plt.grid(True, color='g', linestyle='--', linewidth=1, alpha=0.4)
    
    
