import h5py
import numpy as np
import os.path as path
# import pickle as pkl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
# import itertools
import yaml 
# import corner
import pandas as pd
# import xml.etree.ElementTree as ET
# from scipy.stats import chi2
import analysis

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


def interpolate_uvlf_z(uvlfs, zs, type):
    interp_z = [12.0]
    uvlf_z12 = np.zeros_like(uvlfs[0])
    for i in range(len(uvlfs[0])):
        if type=='logphi':
            uvlf_z12[i] =  np.interp(interp_z, zs, np.log10([uvlfs[0][i],uvlfs[1][i]]))
        elif type=='phi':
            uvlf_z12[i] =  np.interp(interp_z, zs, [uvlfs[0][i],uvlfs[1][i]])
    return uvlf_z12

def interpolate_uvlf_t(uvlfs, zs, type):
    ts = cosmo.age(zs).value[::-1]
    interp_t = [cosmo.age(12.0).value]
    uvlf_z12 = np.zeros_like(uvlfs[0])
    for i in range(len(uvlfs[0])):
        if type=='logphi':
            uvlf_z12[i] =  np.interp(interp_t, ts, np.log10([uvlfs[0][i],uvlfs[1][i]])[::-1])
        elif type=='phi':
            uvlf_z12[i] =  np.interp(interp_t, ts, [uvlfs[0][i],uvlfs[1][i]][::-1])
    return uvlf_z12


# /carnegie/nobackup/users/gdriskell/
base_dir = '/carnegie/nobackup/users/gdriskell/jwst_data/'


bestfit_alphaStar = -1.5
bestfit_timescale = 0.1
bestfit_velocityOutflow = 150
bestfit_alphaOutflow = 1.5
# f, axs = plt.subplots(3,1,figsize=(5,10), constrained_layout=True)
f, axs = plt.subplots(2,3,figsize=(15,8.0), constrained_layout=True) # sharey=True
# taus = [0.1, 0.5, 0.9]
taus = np.linspace(0.01,0.1,10)

# bin_widths = [0.25, 0.5, 1.0, 1.5]
bin_width = 0.75
zs = [8.0,12.0,16.0][::-1]

norm = mpl.colors.Normalize(vmin=0.01, vmax=0.1)
sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')


for i,z in enumerate(zs):
    # ax = axs[i]
    for j in range(10):
        j = 9-j
        data_dir = path.join(base_dir, f'test_lower_tau0_p{j}')
        # print(data_dir)
        # out_fn = path.join(data_dir,f'z{z}.hdf5')
        # outfile = h5py.File(out_fn)
        # logmhs, muvs, weights = get_Muv_from_hdf5(outfile, 'JWST_NIRCAM_f277w')
        # x, uvlf = get_uvlf(muvs, weights, bin_width)

        uvlf = analysis.get_uvlf(None, None, data_dir, True, False, False)/analysis.dabs
        idx = analysis.redshift_grid == z
        # print(uvlf.shape)
        uvlf = uvlf[:,idx]
        x = analysis.absolute_magnitude_grid
        # uvlfs.append(uvlf)
        if j == 9:
            bestfit = np.log10(uvlf)
            axs[0,i].plot(x, np.log10(uvlf), color=sm.to_rgba(taus[j]))
            axs[1,i].plot(x, (np.log10(uvlf)-bestfit)/bestfit * 100, color=sm.to_rgba(taus[j]))
        else:
            # print(j,taus[j])
            axs[0,i].plot(x, np.log10(uvlf), color=sm.to_rgba(taus[j]))
            axs[1,i].plot(x, (np.log10(uvlf)-bestfit)/np.abs(bestfit) * 100, color=sm.to_rgba(taus[j]))
            axs[0,i].set_ylim(-7.5, -1.5)
            axs[1,i].set_ylim(-10, 5)
    
    axs[0,i].set_title(f'$z={z}$', fontsize=22.5)
    axs[1,i].set_xlabel(r'$M_{\mathrm{UV}}$')
    axs[0,i].set_xlim(-23, -16.5)
    axs[1,i].set_xlim(-23, -16.5)
    
    
    # p.legend(frameon=False)
    if i == 0:
        # axs[0,i].set_ylim(-7.5, -1.5)
        # axs[1,i].set_ylim(-0.05, 0.1)
        axs[0,i].set_ylabel(r'$\mathrm{Log}(\phi_{\mathrm{UV}}\,/\,\mathrm{Mpc^{-3} mag^{-1}})$')
        axs[1,i].set_ylabel(r'$\frac{\mathrm{Log}(\phi_{\mathrm{UV}})-\mathrm{Log}(\phi_{\tau=0.1})}{|\mathrm{Log}(\phi_{\tau=0.1})|} [\%]$')
cbar = plt.colorbar(sm,ax=axs, pad = 0.01)
cbar.set_label(r'$\tau_{0}$', fontsize=22.5)
plt.savefig(f'test_lower_tau0_uvlfs.pdf')
plt.close('all')

# data = np.array([x, uvlf])
# print (data.shape)
# np.save('binned_uvlf_data.npy',data)
