"""
Plot test results for lower tau0 parameter analysis.

This module creates visualization plots for the lower tau0 likelihood
analysis, showing how the likelihood varies with tau0 values.
"""

import numpy as np
import os.path as path
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import analysis

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 17.5
mpl.rcParams['ytick.labelsize'] = 17.5
mpl.rcParams['axes.labelsize'] = 22.5
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['figure.titlesize'] = 23
mpl.rcParams['xtick.major.size'] = 8#10
mpl.rcParams['ytick.major.size'] = 8#10
mpl.rcParams['xtick.minor.size'] = 6
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'


base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data'

f, axs = plt.subplots(2,3,figsize=(15,8.0), constrained_layout=True) 
taus = np.linspace(0.01,0.1,10)

zs = [8.0,12.0,16.0][::-1]

norm = mpl.colors.Normalize(vmin=0.01, vmax=0.1)
sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')


for i,z in enumerate(zs):
    for j in range(10):
        j = 9-j
        data_dir = path.join(base_dir, f'test_lower_tau0_p{j}')

        uvlf = analysis.get_uvlf(None, None, data_dir, True, False, False)/analysis.dabs
        idx = analysis.redshift_grid == z
        uvlf = uvlf[:,idx]
        x = analysis.absolute_magnitude_grid

        if j == 9:
            bestfit = np.log10(uvlf)
        
        axs[0,i].plot(x, np.log10(uvlf), color=sm.to_rgba(taus[j]))
        axs[1,i].plot(x, (np.log10(uvlf)-bestfit)/np.abs(bestfit) * 100, color=sm.to_rgba(taus[j]))
            
    
    axs[0,i].set_title(f'$z={z}$', fontsize=22.5)
    axs[1,i].set_xlabel(r'$M_{\mathrm{UV}}$')
    axs[0,i].set_xlim(-23, -16.5)
    axs[1,i].set_xlim(-23, -16.5)
    axs[0,i].set_ylim(-7.5, -1.5)
    axs[1,i].set_ylim(-10, 5)
    
    if i == 0:
        axs[0,i].set_ylabel(r'$\mathrm{Log}(\phi_{\mathrm{UV}}\,/\,\mathrm{Mpc^{-3} mag^{-1}})$')
        axs[1,i].set_ylabel(r'$\frac{\mathrm{Log}(\phi_{\mathrm{UV}})-\mathrm{Log}(\phi_{\tau=0.1})}{|\mathrm{Log}(\phi_{\tau=0.1})|} [\%]$')
cbar = plt.colorbar(sm,ax=axs, pad = 0.01)
cbar.set_label(r'$\tau_{0}$', fontsize=22.5)
plt.savefig(f'test_lower_tau0_uvlfs.pdf')
plt.close('all')

