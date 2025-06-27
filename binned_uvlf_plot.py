import h5py
import numpy as np
import os.path as path
import pickle as pkl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import itertools
import yaml 
# import corner
import pandas as pd
import xml.etree.ElementTree as ET
import analysis
# from scipy.stats import chi2

cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)

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


def get_Muv_from_hdf5(outfile, filter):
    outputs = outfile['Outputs']    
    nodeData = outputs['Output1']['nodeData']
    Mhs = nodeData['basicMass'][:]
    treeWeights = nodeData['mergerTreeWeight'][:]

    z = nodeData['redshift'][0]
    sfn = 'spheroidLuminositiesStellar:{}:observed:z{:.4f}'.format(filter, z)
    dfn = 'diskLuminositiesStellar:{}:observed:z{:.4f}'.format(filter, z)
    Lum = nodeData[sfn][:]+nodeData[dfn][:] # changed
    Lum[Lum<1.0e-5] = 1.0e-5 
    abs_mag = -2.5*np.log10(Lum)
    app_mag = abs_mag+cosmo.distmod(z).value-2.5*np.log10(1+z)
    sortIdx = np.argsort(Mhs)
    Mhs = Mhs[sortIdx]
    app_mag = app_mag[sortIdx]
    abs_mag = abs_mag[sortIdx]
    treeWeights = treeWeights[sortIdx]
    # data = np.stack((np.log10(Mhs), app_mag, abs_mag),axis=-1)
    return np.log10(Mhs), abs_mag, treeWeights 


def get_uvlf(muvs, weights, bin_width):
    dmuv = bin_width
    bins = np.arange(-24, -14, dmuv)
    bin_centers = bins[:-1]+dmuv/2.0
    left = bins[:-1]
    right = bins[1:]
    uvlf = np.zeros_like(bin_centers)
    for i in range(len(bins)-1):
        idx = (muvs < right[i]) & (muvs >= left[i])
        uvlf[i] = np.sum(weights[idx])/dmuv
    # uvlf = np.log10(uvlf)
    return bin_centers, uvlf


# def cutoff_pdf():
#     for i in range(analysis.nz):
#         for j in range(analysis.n_mass_bins):
            
#     return

# /carnegie/nobackup/users/gdriskell/
base_dir = '/carnegie/nobackup/users/gdriskell/jwst_data/'


# f, axs = plt.subplots(3,1,figsize=(5,10), constrained_layout=True)
f, axs = plt.subplots(1,3,figsize=(12.5,4.0), constrained_layout=True, sharey=True)
# taus = [0.1, 0.5, 0.9]
taus = np.linspace(0.01,0.1,10)

# bin_widths = [0.25, 0.5, 1.0, 1.5]
bin_width = 0.75 
zs = [8.0,12.0,16.0][::-1]


for i,z in enumerate(zs):
    ax = axs[i]
    # for j in range(10):
    # data_dir = path.join(base_dir, 'paper_params_p13845')
    data_dir = path.join(base_dir, 'paper_params_p9380')
    out_fn = path.join(data_dir, f'z{z}.hdf5')
    outfile = h5py.File(out_fn)
    logmhs, muvs, weights = get_Muv_from_hdf5(outfile, 'JWST_NIRCAM_f277w')

    uvlf_fn = path.join(data_dir, 'absolute_uvlf.npy')
    load_uvlf = np.load(uvlf_fn)

    # skew_uvlf_fn = path.join(data_dir, 'skewed_absolute_uvlf.npy')
    # skew_uvlf = np.load(skew_uvlf_fn)
    skew_probs = analysis.get_skewed_probs(analysis.absolute_magnitude_grid, None, data_dir, True, False, False)

    skew_uvlf = analysis.get_uvlf(
        skew_probs,
        analysis.binned_weights,
        data_dir, 
        True, 
        True,
        True,
    )

    zidx = analysis.redshift_grid == z
    load_uvlf = load_uvlf[:,zidx]
    skew_uvlf = skew_uvlf[:,zidx]
    x, uvlf = get_uvlf(muvs, weights, bin_width)
    # uvlfs.append(uvlf)
    ax.plot(analysis.absolute_magnitude_grid, np.log10(load_uvlf/analysis.dabs), label='Gaussian Model')
    ax.plot(x, np.log10(uvlf), label='Simulation')
    ax.plot(analysis.absolute_magnitude_grid, np.log10(skew_uvlf/analysis.dabs), '--', color='r', label='Skewed Model (truncated)')
    
    ax.set_title(f'$z={z}$', fontsize=22.5)
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_xlim(-23, -16.5)
    ax.set_ylim(-7.5, -1.5)
    ax.legend(frameon=False)
    if i == 0:
        ax.set_ylabel(r'$\mathrm{Log}(\phi_{\mathrm{UV}}\,/\,\mathrm{Mpc^{-3} mag^{-1}})$')

# cbar = plt.colorbar(sm,ax=ax)
# cbar.set_label(r'$\tau_{0}$', fontsize=22.5)
plt.savefig(f'test_uvlf_truncated.pdf')
# plt.savefig(f'test_uvlf.pdf')
plt.close('all')

# data = np.array([x, uvlf])
# print (data.shape)
# np.save('binned_uvlf_data.npy',data)
