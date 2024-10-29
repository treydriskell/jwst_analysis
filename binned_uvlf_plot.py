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
import cmasher as cmr
# from scipy.stats import chi2

cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
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


def get_uvlf(muvs, weights):
    dmuv = 0.5
    bins = np.arange(-23, -17, dmuv)
    bin_centers = bins[:-1]+dmuv/2.0
    left = bins[:-1]
    right = bins[1:]
    uvlf = np.zeros_like(bin_centers)
    for i in range(len(bins)-1):
        idx = (muvs < right[i]) & (muvs >= left[i])
        uvlf[i] = np.sum(weights[idx])#/dmh
    # uvlf = np.log10(uvlf)
    return bin_centers, uvlf


base_dir = '/scratch1/gdriskel/jwst_data/'

bestfit_alphaStar = -1.5
bestfit_timescale = 0.1
bestfit_velocityOutflow = 150
bestfit_alphaOutflow = 1.5

f = plt.figure()

data_dir = path.join(base_dir, 'test_params_p0')
out_fn = path.join(data_dir,'z16.0.hdf5')
outfile = h5py.File(out_fn)
logmhs, muvs, weights = get_Muv_from_hdf5(outfile, 'JWST_NIRCAM_f277w')
x, uvlf = get_uvlf(muvs, weights)

plt.plot(x, np.log10(uvlf), 'k-')
plt.xlabel(r'$M_{\mathrm{UV}}$')
plt.ylabel(r'$\mathrm{Log}(\phi_{\mathrm{UV}}\,/\,\mathrm{Mpc^{-3} mag^{-1}})$')
plt.xlim(-23, -16.5)
plt.ylim(-6.5, -1.0)
plt.tight_layout()
plt.savefig(f'binned_bestfit_uvlf.pdf')
plt.close('all')



