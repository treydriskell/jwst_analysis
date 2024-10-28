
import h5py
import numpy as np
import os
import os.path as path
import pickle as pkl
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM
import itertools
# import glob
import pandas as pd
# import cmasher as cmr
import seaborn as sns

rng = np.random.default_rng()

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['xtick.major.size'] = 5#10
mpl.rcParams['ytick.major.size'] = 5#10
mpl.rcParams['xtick.minor.size'] = 6
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'

# global cosmo
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)


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
    data = np.stack((np.log10(Mhs), app_mag, abs_mag),axis=-1)
    return data, treeWeights 


def binned_uvlf(weights, Muvs):
    min_Muv = -23 #np.amin(Muvs)
    max_Muv = -10 # np.amax(Muvs)
    dMuv = 0.5
    nbins = int((max_Muv-min_Muv)/dMuv)
    Muv_bins = np.linspace(min_Muv, max_Muv, nbins+1)
    # dMuv = (max_Muv-min_Muv) / float(nbin)
    # print(dMuv)
    uvlf = []
    for j in range(nbins):
        idx = np.argwhere((Muv_bins[j] < Muvs) & (Muvs < Muv_bins[j+1]))
        uvlf.append(np.sum(weights[idx]))
    uvlf = np.array(uvlf)/dMuv
    x = Muv_bins[:-1]+dMuv/2.0
    # out = np.stack((x, uvlf)).T
    return x, uvlf


def compute_uvlf(weights, probs, data_dir, output_tag, recompute=False, abs_mag=False):
    # Output shape is muv x z
    if abs_mag:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf_abs.npy')
    else:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf.npy')
    if path.isfile(uvlf_fn) and not recompute:
        phi = np.load(uvlf_fn)
    # else:
        # raise Exception()
        # npad = weights.shape[-1]-probs.shape[-1]
        # probs = np.pad(probs, [[0,0],[0,0],[npad,0]])
        # # probs = probs.T
        # # probs = (probs / np.sum(probs, axis=0))
        # # probs = probs.T
        # probs = (probs / np.sum(probs, axis=0))
        # probs = np.where(np.isnan(probs), 0.0, probs)
        # phi = np.sum(weights * probs, axis=2)
        # np.save(uvlf_fn, phi)
    return phi


def plot_uvlf(axs, muv_grid, z_grid, uvlf, data_dir, abs=False):
    c = 'k' #s_m.to_rgba(i)
    # c = plt.cm.Dark2(3)
    # zs = [8.0, 12.0, 16.0]
    # for i,z in enumerate(zs):
    ax = axs[0]
    zidx = (z_grid > 8.5) & (z_grid < 9.5)
    # if abs:
    #     abs_grid = muv_grid
    # else:
    #     abs_grid = muv_grid - cosmo.distmod(9.0).value+2.5*np.log10(1+9.0)
    #     dm = (abs_grid[2:]-abs_grid[:-2])/2.0
    # print(np.sum(zidx))
    y = np.average(uvlf[:, zidx],axis=1) 
    # y = uvlf[:, zidx][:,-1]
    # print(np.amin(y), np.amax(y))
    midx = y>0
    y = np.log10(y[midx]) # /dm[midx]
    ax.plot(abs_grid[midx], y, lw=2.0, c=c, label='Best fit') # label='Sim.',
    ax = axs[1]
    zidx = (z_grid > 9.5) & (z_grid < 12.0)
    # if abs:
    #     abs_grid = muv_grid
    #     dm = abs_grid[1]-abs_grid[0]
    # else:
    #     abs_grid = muv_grid - cosmo.distmod(10.5).value+2.5*np.log10(1+10.5)
    #     # dm = np.array([abs_grid[1]-abs_grid[0], abs_grid[1:]-abs_grid[:-1], abs_grid[-1]-abs_grid[-2]])
    #     dm = ((abs_grid[1:-1]-abs_grid[:-2]) + (abs_grid[2:]-abs_grid[1:-1]))/2.0
    y = np.average(uvlf[:, zidx],axis=1)#[1:-1]
    midx = y>0
    y = np.log10(y[midx]) # /dm[midx]
    ax.plot(abs_grid[midx], y, lw=2.0, c=c, label='Best fit') #  label='Sim.',
    

def plot_data(axs):
    ax = axs[0]
    ngdeep_muv = [-20.1, -19.1, -18.35, -17.85, -17.35]
    ngdeep_phi = np.array([14.7e-5, 18.9e-5, 74.0e-5, 170.0e-5, 519.0e-5])
    ngdeep_phi_err = np.array([[7.2e-5, 8.9e-5, 29.0e-5, 65.e-5, 198.e-5],
                      [11.1e-5, 13.8e-5, 41.4e-5, 85.e-5, 248.e-5]])
    ngdeep_log_phi = np.log10(ngdeep_phi)
    ngdeep_log_err = [ngdeep_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-ngdeep_log_phi]
    ngdeep_log_err2 = [ngdeep_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]*2),np.log10(ngdeep_phi+ngdeep_phi_err[1,:]*2)-ngdeep_log_phi]
    
    ceers_muv_z9 = [-22.0, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5]
    ceers_phi_z9 = np.array([1.1e-5, 2.2e-5, 8.2e-5, 9.6e-5, 28.6e-5, 26.8e-5, 136.0e-5])
    ceers_log_phi = np.log10(ceers_phi_z9)
    ceers_upper_z9 = np.array([0.7e-5, 1.3e-5, 4.0e-5, 4.6e-5, 11.5e-5, 12.4e-5, 61.0e-5])
    ceers_lower_z9 = np.array([0.6e-5, 1.0e-5, 3.2e-5, 3.6e-5, 9.1e-5, 10.0e-5,49.9e-5])
    ceers_log_err = [ceers_log_phi-np.log10(ceers_phi_z9-ceers_lower_z9),np.log10(ceers_phi_z9+ceers_upper_z9)-ceers_log_phi]
    ceers_log_lower_err2 = ceers_log_phi-np.log10(ceers_phi_z9-ceers_lower_z9*2)
    ceers_log_upper_err2 = np.log10(ceers_phi_z9+ceers_upper_z9*2)-ceers_log_phi

    ceers_log_err2 = [ceers_log_lower_err2[1:], ceers_log_upper_err2[1:]]
    
    ax.scatter(ngdeep_muv, ngdeep_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',  label='NGDEEP (Leung et. al. 2023)')
    ax.errorbar(ngdeep_muv, ngdeep_log_phi, elinewidth=1.8, yerr=ngdeep_log_err, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    # ax.errorbar(ngdeep_muv, ngdeep_log_phi, yerr=ngdeep_log_err2, 
    #             elinewidth=1.8,  marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar([-21.1],[np.log10(8.9e-5)], yerr=ngdeep_log_err[0][0]/2., uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z9, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    ax.errorbar(ceers_muv_z9, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    # ax.errorbar(ceers_muv_z9[1:], ceers_log_phi[1:], yerr=ceers_log_err2,
    #             marker='o', color="none", ecolor='gray', capsize=3.5, ls='none',elinewidth=1.8) #  
    # ax.errorbar(ceers_muv_z9[0], ceers_log_phi[0], yerr=[[ceers_log_err[0][0]],[ceers_log_upper_err2[0]]], #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8) # ls='none'
    ax.errorbar([-22.5, -21.5], [np.log10(0.9e-5),np.log10(0.9e-5)], yerr=ceers_log_err[0][0]/2., uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 
    
    ax = axs[1]
    ngdeep_muv = [-19.35, -18.65, -17.95, -17.25]
    ngdeep_phi = np.array([18.5e-5, 27.7e-5, 59.1e-5, 269.0e-5])
    ngdeep_phi_err = np.array([[8.3e-5, 13.0e-5, 29.3e-5, 124.e-5],
                    [11.9e-5, 18.3e-5, 41.9e-5, 166.e-5]])
    obs_log_phi = np.log10(ngdeep_phi)
    log_err = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-obs_log_phi]
    log_err2 = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]*2),np.log10(ngdeep_phi+ngdeep_phi_err[1,:]*2)-obs_log_phi]

    ceers_muv_z11 = [-20.5, -20.0, -19.5, -19.0, -18.5]
    ceers_phi_z11 = np.array([1.8e-5, 5.4e-5, 7.6e-5, 17.6e-5, 26.3e-5])
    ceers_log_phi = np.log10(ceers_phi_z11)
    ceers_upper_z11 = np.array([1.2e-5, 2.7e-5, 3.9e-5, 10.3e-5, 18.2e-5])
    ceers_lower_z11 = np.array([0.9e-5, 2.1e-5, 3.0e-5, 7.9e-5, 13.3e-5])
    ceers_log_err = [ceers_log_phi-np.log10(ceers_phi_z11-ceers_lower_z11),np.log10(ceers_phi_z11+ceers_upper_z11)-ceers_log_phi]
    ceers_log_err2 = [ceers_log_phi-np.log10(ceers_phi_z11-ceers_lower_z11*2),np.log10(ceers_phi_z11+ceers_upper_z11*2)-ceers_log_phi]
    ceers_log_err2[0]=ceers_log_err2[0][1:][:-1]
    ceers_log_err2[1]=ceers_log_err2[1][1:][:-1]

    ax.scatter(ngdeep_muv, obs_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',label='NGDEEP data (Leung et. al. 2023)',)
    ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    # ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err2, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar([-20.05],[np.log10(9.7e-5)], yerr=log_err[0][0]/2., uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z11, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    ax.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    # ax.errorbar(ceers_muv_z11[1:][:-1], ceers_log_phi[1:][:-1], elinewidth=1.8, yerr=ceers_log_err2, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    # ax.errorbar([ceers_muv_z11[0],ceers_muv_z11[-1]], [ceers_log_phi[0],ceers_log_phi[-1]], yerr=[[ceers_log_err[0][0],ceers_log_err[0][-1]],[ceers_log_err2[1][0],ceers_log_err2[1][-1]]], #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') # ls='none'
    ax.errorbar([-21.0], [np.log10(0.5e-5)], yerr=ceers_log_err[0][0]/2., uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 
    
    
def load_data(data_dir):
    data_fn = path.join(data_dir, 'data.npy')
    if False: #path.isfile(data_fn):
        data = np.load(data_fn)
    else:
        app_mags = []
        abs_mags = []
        logmhs = []
        zs = []
        for z in ['8.0', '12.0', '16.0']: #
            print(z)
            #outfiles = [data_dir+f'z{z}:MPI{i:04d}.hdf5' for i in range(160)] #glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            # outfiles = glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            mhs = []
            app = []
            abs = []
            weights = []
            fn = data_dir+f'z{z}.hdf5'
            # for fn in outfiles:
            try: 
                f = h5py.File(fn,"r")
                if 'Outputs' in f:
                    d,w = get_Muv_from_hdf5(f, 'JWST_NIRCAM_f277w')
                    mhs.append(d[:,0].flatten())
                    # app.append(d[:,1].flatten())
                    abs.append(d[:,2].flatten())
                    weights.append(w.flatten())
                    # print(len(mhs[-1]))
                f.close()
            except:
                continue
            mhs = np.concatenate(mhs)
            idx = np.argsort(mhs)
            mhs = mhs[idx]
            # app = np.concatenate(app)[idx]
            abs = np.concatenate(abs)[idx]
            logmhs.append(mhs)
            # app_mags.append(app)
            abs_mags.append(abs)
            zs.extend([float(z) for i in range(len(logmhs[-1]))])
        
        # app_mags = np.concatenate(app_mags)
        weights = np.concatenate(weights)
        abs_mags = np.concatenate(abs_mags)
        # logmhs = np.concatenate(logmhs)
        zs = np.array(zs)
        # data = np.stack([logmhs, zs, app_mags, abs_mags], axis=1)
        # np.save(data_fn, data)
    return weights, abs_mags, zs


# def get_ngdeep_cf(app_grid):
#     """z~9 covers z=8.5-9.5 and z~11 covers z=9.5-12.0
#     Can't find survey area so just dividing by max for now"""
#     z9_volume = cosmo.comoving_volume(9.5)-cosmo.comoving_volume(8.5)
#     z11_volume = cosmo.comoving_volume(12.0)-cosmo.comoving_volume(9.5)
#     ngdeep_absmag_z9 = np.array([-21.1, -20.1, -19.1, -18.35, -17.85, -17.35])
#     ngdeep_appmag_z9 = ngdeep_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
#     ngdeep_veff_z9 = np.array([18700., 18500., 15800., 13100., 7770., 2520.]) # Mpc^3
#     # ngdeep_absmag_z11 = np.array([-20.05, -19.35, -18.65, -17.95, -17.25, -17.05]) # last value using extrapolation
#     # ngdeep_veff_z11 = np.array([27900., 26100., 20800., 9840., 2210., 0.]) 
  
#     # ngdeep_appmag_z11 = ngdeep_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
#     ngdeep_z9_cf = ngdeep_veff_z9 / np.amax(ngdeep_veff_z9) #z9_volume
#     # ngdeep_z11_cf = ngdeep_veff_z11 / np.amax(ngdeep_veff_z11) #z11_volume

#     ### TODO: really dumb version for now <------!
#     # print(ngdeep_appmag_z9)
#     # print(ngdeep_appmag_z11)
#     # ngd_cf = np.concatenate(([1.0], ngdeep_z11_cf, [0.0]))
#     # ngd_app = np.concatenate(([np.amin(app_grid)], ngdeep_appmag_z11, [np.amax(app_grid)]))
#     ngdeep_appmag_z9 = np.append(ngdeep_appmag_z9, 30.4)
#     ngdeep_z9_cf = np.append(ngdeep_z9_cf, 0.0)
    
#     ngd_cf = np.concatenate(([1.0], ngdeep_z9_cf, [0.0]))
#     ngd_app = np.concatenate(([np.amin(app_grid)], ngdeep_appmag_z9, [np.amax(app_grid)]))
#     cf = np.interp(app_grid, ngd_app, ngd_cf)
#     return cf


# def get_ceers_cf(app_grid):
#     """z~9 covers z=8.5-9.7 and z~11 covers z=9.7-13.0, z~14 covers > 13
#     Can't find survey area so just dividing by max for now"""
#     z9_volume = cosmo.comoving_volume(9.7)-cosmo.comoving_volume(8.5)
#     # z11_volume = cosmo.comoving_volume(13.0)-cosmo.comoving_volume(9.7)
#     # z14_volume = cosmo.comoving_volume(14.5)-cosmo.comoving_volume(13.0)
#     ceers_absmag_z9 = np.array([-22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5])
#     ceers_appmag_z9 = ceers_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
#     ceers_veff_z9 = np.array([187000., 187000., 187000., 193000., 177000., 161000., 120000., 77900., 18600.]) # Mpc^3
    
#     # ceers_absmag_z11 = np.array([-21.0, -20.5, -20.0, -19.5, -19.0, -18.5, -18.0]) # last value using extrapolation
#     # ceers_veff_z11 = np.array([373000., 354000., 306000., 220000., 62100., 23800., 0.0]) 
#     # ceers_appmag_z11 = ceers_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
    
#     # ceers_absmag_z14 = np.array([-20.5, -20.0, -19.5, -19.0,]) # last value using extrapolation
#     # ceers_veff_z14 = np.array([147000., 85400., 60600., 0.0]) 
#     # ceers_appmag_z14 = ceers_absmag_z11+cosmo.distmod(14.0).value-2.5*np.log10(1+14.0)

#     ceers_z9_cf = ceers_veff_z9 / np.amax(ceers_veff_z9) #z9_volume
#     # ceers_z11_cf = ceers_veff_z11 / np.amax(ceers_veff_z11) #z11_volume
#     # ceers_z14_cf = ceers_veff_z14 / np.amax(ceers_veff_z14)

#     ceers_appmag_z9 = np.append(ceers_appmag_z9, 29.15)
#     ceers_z9_cf = np.append(ceers_z9_cf, 0.0)
#     # print(ceers_appmag_z9)
#     ceers_cf = np.concatenate(([1.0], ceers_z9_cf, [0.0]))
#     ceers_app = np.concatenate(([np.amin(app_grid)], ceers_appmag_z9, [np.amax(app_grid)]))
#     cf = np.interp(app_grid, ceers_app, ceers_cf)
#     # print(cf)
#     return cf



app_cutoff = 30.4

# global z_grid
# global s 
dMh = 0.1
Mh_bins = np.arange(8.0, 11.76, dMh)
global nm
nm = len(Mh_bins)-1
bin_centers = Mh_bins[:-1] + dMh/2.0 

global nz
nz = 17
global z_grid
z_grid = np.linspace(8.0, 16.0, nz)
dz = z_grid[1]-z_grid[0]

z_left = z_grid-dz/2.0
z_right = z_grid+dz/2.0

logmhs = np.log10(np.geomspace(1.0e8,5.0e11,3699))
# global binned_weights
# binned_weights = get_binned_weights(Mh_bins,logmhs,zgrid_weights)
# print(binned_weights)

abs_min = -25.0
abs_max = 0.
dabs = 0.2
nabs = int(round((abs_max-abs_min)/dabs))+1
abs_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = abs_grid[1]-abs_grid[0]

app_min = 22.0
app_max = 45.0
dapp = 0.2
napp = int(round((app_max-app_min)/dapp))+1
app_grid = np.linspace(app_min, app_max, napp) 
dapp = app_grid[1]-app_grid[0]


z_volumes = (cosmo.comoving_volume(z_grid+dz/2.0)-cosmo.comoving_volume(z_grid-dz/2.0)).value 

# ngdeep_cf = get_ngdeep_cf(app_grid)
# ngdeep_area = 3.3667617763401435e-08 # 5 arcmin^2 as a fraction of the sky
# # print(ngdeep_area)
# ngdeep_volumes = ngdeep_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
# ngdeep_eff_volume = ngdeep_cf.reshape(-1,1)*ngdeep_volumes.reshape(1,-1)
# ngdeep_eff_volume = np.abs(ngdeep_eff_volume)

# ceers_cf = get_ceers_cf(app_grid)
# ceers_area = 5.932234249911333e-07 # 88.1 arcmin^2 as a fraction of the sky
# # print(ceers_area)
# ceers_volumes = ceers_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
# ceers_eff_volume = ceers_cf.reshape(-1,1)*ceers_volumes.reshape(1,-1)
# ceers_eff_volume = np.abs(ceers_eff_volume)

# volumes = ceers_eff_volume+ngdeep_eff_volume

output_tag = 'interp'

f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)
plot_data(axs)

# fn = '/data001/gdriskell/jwst_blitz_astro_samples/final_params_p1609/'
# uvlf = compute_uvlf(None, None, fn, output_tag, False, True)/dabs
# plot_uvlf(axs, abs_grid, z_grid, uvlf, fn, True)


z9_upper = np.zeros_like(abs_grid)
z9_lower = np.ones_like(abs_grid)
z11_upper = np.zeros_like(abs_grid)
z11_lower = np.ones_like(abs_grid)

# def get_i(a,b,c,d):
#     return a+2*b+4*c+8*d
# ns = list(range(16))

base_dir = '/data001/gdriskell/jwst_blitz_astro_samples/'
df = pd.read_csv('final_df.csv')


fn = base_dir+f'nr4_p60/'
# print(fn)
uvlf = compute_uvlf(None, None, fn, output_tag, False, True)/dabs
zidx = (z_left > 8.5) & (z_right < 9.5)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
bestfit_z9 = y
axs[0].plot(abs_grid, np.log10(y), '-k', label='Best fit')


zidx = (z_left > 9.5) & (z_right < 12.0)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        # if j==0:
            # print(right_cutoff)
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
bestfit_z11 = y
axs[1].plot(abs_grid, np.log10(y), '-k', label='Best fit')


# n_samples = [100,1000]
n_sample = 5000 # 5000
# fns = [f'/data001/gdriskell/jwst_blitz_astro_samples/final_corners_p{i}' for i in ns]
colors = sns.color_palette("Blues",n_colors=2) 

# for i, n in enumerate(n_samples):
z9_upper = np.zeros_like(abs_grid)
z9_lower = np.ones_like(abs_grid)
z11_upper = np.zeros_like(abs_grid)
z11_lower = np.ones_like(abs_grid)
# c = colors[-(i+1)]
# c = colors[-(i+1)]
samples = df.sample(n_sample, replace=True, weights='like')
z9_ys = []
z11_ys = []
for index,row in samples.iterrows():
    fn = base_dir+f'{row["tag"]}_p{row["idx"]}/'
    # print(fn)
    uvlf = compute_uvlf(None, None, fn, output_tag, False, True)/dabs
    zidx = (z_left > 8.5) & (z_right < 9.5)
    zs = z_grid[zidx]
    z_left_slice = z_left[zidx]
    z_right_slice = z_right[zidx]
    zuvlf = uvlf[:,zidx]
    zvolume = z_volumes[zidx]
    totalV = np.zeros_like(abs_grid)
    y = np.zeros_like(abs_grid)
    for j,muv in enumerate(abs_grid):
        for k in range(len(zs)):
            # left > right
            # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
            right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
            if muv < right_cutoff:
                y[j] += zuvlf[j,k]*zvolume[k]
                totalV[j] += zvolume[k]
            # elif muv < left_cutoff:
            #     volume = 
    y /= totalV

    # z9_upper = np.amax([z9_upper,y],axis=0)
    # z9_lower = np.amin([z9_lower,y],axis=0)
    z9_ys.append(y)

    zidx = (z_left > 9.5) & (z_right < 12.0)
    zs = z_grid[zidx]
    z_left_slice = z_left[zidx]
    z_right_slice = z_right[zidx]
    zuvlf = uvlf[:,zidx]
    zvolume = z_volumes[zidx]
    totalV = np.zeros_like(abs_grid)
    y = np.zeros_like(abs_grid)
    for j,muv in enumerate(abs_grid):
        for k in range(len(zs)):
            # left > right
            # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
            right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
            # if j==0:
                # print(right_cutoff)
            if muv < right_cutoff:
                y[j] += zuvlf[j,k]*zvolume[k]
                totalV[j] += zvolume[k]
            # elif muv < left_cutoff:
            #     volume = 
    y /= totalV
            
    # z11_upper = np.amax([z11_upper,y],axis=0)
    # z11_lower = np.amin([z11_lower,y],axis=0)
    z11_ys.append(y)

z9_ys = np.array(z9_ys)
z11_ys = np.array(z11_ys)
# print(z9_ys.shape)

z9_diff = np.abs(z9_ys-bestfit_z9) 
z11_diff = np.abs(z11_ys-bestfit_z11) 
# print(z9_diff.shape)

# oneSigmaScatter = []
# twoSigmaScatter = []

for i,frac in enumerate([0.68, 0.95]):
    z11_lower = np.zeros_like(abs_grid)
    z9_lower = np.zeros_like(abs_grid)
    z11_upper = np.zeros_like(abs_grid)
    z9_upper = np.zeros_like(abs_grid)
    c = colors[-(i+1)]
    n_cut = int(round(frac*n_sample))
    for im, m in enumerate(abs_grid):
        z9_idx = np.argsort(z9_diff[:,im])
        z11_idx = np.argsort(z11_diff[:,im])

        z9_diff_m = z9_diff[z9_idx,im]
        z11_diff_m = z11_diff[z11_idx,im]
        z9_ys_m = z9_ys[z9_idx,im]
        z11_ys_m = z11_ys[z11_idx,im]

        cut_uvlfs_z9 = z9_ys_m[:n_cut]
        z9_upper[im] = np.amax(cut_uvlfs_z9, axis=0)
        z9_lower[im] = np.amin(cut_uvlfs_z9, axis=0)

        cut_uvlfs_z11 = z11_ys_m[:n_cut]
        z11_upper[im] = np.amax(cut_uvlfs_z11, axis=0)
        z11_lower[im] = np.amin(cut_uvlfs_z11, axis=0)

    idx = z9_lower > 0
    z9_upper = np.log10(z9_upper[idx])
    z9_lower = np.log10(z9_lower[idx])
    ax = axs[0]
    # print(idx.shape)
    ax.fill_between(abs_grid[idx], z9_lower, z9_upper, color=c, alpha=0.9-i*0.2, zorder=-i, 
                    label=f'${int(round(frac*100))}'+r'\% $ CI')
    idx1 = idx
    
    idx = z11_lower > 0
    z11_upper = np.log10(z11_upper[idx])
    z11_lower = np.log10(z11_lower[idx])
    ax = axs[1]
    ax.fill_between(abs_grid[idx], z11_lower, z11_upper, color=c, alpha=0.9-i*0.2, zorder=-i, 
                    label=f'${int(round(frac*100))}'+r'\% $ CI')

    if i == 0:
        oneSigmaScatter = [[abs_grid[idx1],z9_upper-z9_lower], [abs_grid[idx], z11_upper-z11_lower]]
    if i == 1:
        twoSigmaScatter = [[abs_grid[idx1],z9_upper-z9_lower], [abs_grid[idx], z11_upper-z11_lower]]


idx1 = (df['alphaOutflow'] == 0.875) & (df['velocityOutflow'] == 300) & (df['timescale'] == 0.1) & (df['alphaStar'] == -1.5)
row1 = df[idx1]
fn1 = base_dir+f'{row1["tag"].values[0]}_p{row1["idx"].values[0]}/'
print(fn1)
uvlf = compute_uvlf(None, None, fn1, output_tag, False, True)/dabs
zidx = (z_left > 8.5) & (z_right < 9.5)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
# axs[0].plot(abs_grid, np.log10(y), '--k', label=r'Low $\alpha_{\mathrm{outflow}}$, High $V_{\mathrm{outflow}}$')

zidx = (z_left > 9.5) & (z_right < 12.0)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        # if j==0:
            # print(right_cutoff)
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
# axs[1].plot(abs_grid, np.log10(y), '--k', label=r'Low $\alpha_{\mathrm{outflow}}$, High $V_{\mathrm{outflow}}$')

idx2 = (df['alphaOutflow'] == 2.25) & (df['velocityOutflow'] == 100) & (df['timescale'] == 0.1) & (df['alphaStar'] ==-1.5)
row2 = df[idx2]
fn2 = base_dir+f'{row2["tag"].values[0]}_p{row2["idx"].values[0]}/'
uvlf = compute_uvlf(None, None, fn2, output_tag, False, True)/dabs
zidx = (z_left > 8.5) & (z_right < 9.5)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
# axs[0].plot(abs_grid, np.log10(y), ':k', label=r'High $\alpha_{\mathrm{outflow}}$, Low $V_{\mathrm{outflow}}$')

zidx = (z_left > 9.5) & (z_right < 12.0)
zs = z_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = z_volumes[zidx]
totalV = np.zeros_like(abs_grid)
y = np.zeros_like(abs_grid)
for j,muv in enumerate(abs_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        # if j==0:
            # print(right_cutoff)
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
# axs[1].plot(abs_grid, np.log10(y), ':k', label=r'High $\alpha_{\mathrm{outflow}}$, Low $V_{\mathrm{outflow}}$')

# print(z9_lower, z9_upper)
# idx = z9_lower > 0
# z9_upper = np.log10(z9_upper[idx])
# z9_lower = np.log10(z9_lower[idx])
# ax = axs[0]
# # print(z9_lower, z9_upper)
# ax.fill_between(abs_grid[idx], z9_lower, z9_upper, color=c, alpha=0.9-i*0.2, zorder=-i, label=f'N={n}')

# idx = z11_lower > 0
# z11_upper = np.log10(z11_upper[idx])
# z11_lower = np.log10(z11_lower[idx])
# ax = axs[1]
# ax.fill_between(abs_grid[idx], z11_lower, z11_upper, color=c, alpha=0.9-i*0.2, zorder=-i,label=f'N={n}')


ax = axs[0]
ax.set_title(r'$8.5<z<9.5$')
ax.set_xlabel(r'$M_{\mathrm{UV}}$')
ax.set_xlim(-22.75, -17.1)
ax.set_ylim(-6.0, -2.0)
ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\mathrm{[Mpc^{-3} dex^{-1}]}\right)$')
ax.legend(frameon=False, fontsize=12.5, loc='upper left')
ax.set_yticks([-6,-5,-4,-3,-2])

ax = axs[1]
ax.set_title(r'$9.5<z<12.0$')
ax.set_xlabel(r'$M_{\mathrm{UV}}$')
# ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
ax.legend(frameon=False, fontsize=12.5, loc='upper left')
ax.set_xlim(-22.75, -17.1)
ax.set_ylim(-6.0, -2.0)
ax.set_yticks([-6,-5,-4,-3,-2])
# plt.savefig(path.join(data_dir, 'test_model_uvlf.png'), dpi=600)
# plt.savefig(f'sampled_uvlf_n{n_samples}.pdf')
plt.savefig(f'sampled_uvlf_CI.pdf')
# plt.savefig(f'bracketed_uvlf.pdf')
plt.close('all')


f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)

axs[0].plot(oneSigmaScatter[0][0], oneSigmaScatter[0][1], 'k-', label='One Sigma')
axs[0].plot(twoSigmaScatter[0][0], twoSigmaScatter[0][1], 'k--', label='Two Sigma')

axs[1].plot(oneSigmaScatter[1][0], oneSigmaScatter[1][1], 'k-', label='One Sigma')
axs[1].plot(twoSigmaScatter[1][0], twoSigmaScatter[1][1], 'k--', label='Two Sigma')
axs[0].set_title(r'$8.5<z<9.5$')
axs[1].set_title(r'$9.5<z<12.0$')
axs[0].set_xlim(-22.75, -17.1)
axs[1].set_xlim(-22.75, -17.1)
axs[0].set_ylim(0, 1)
axs[0].set_ylabel('Log UVLF Scatter (dex)')
axs[0].legend(frameon=False, fontsize=12.5)
axs[1].legend(frameon=False, fontsize=12.5)
axs[0].set_xlabel(r'$M_{\mathrm{UV}}$')
axs[1].set_xlabel(r'$M_{\mathrm{UV}}$')
plt.savefig(f'uvlf_scatter.pdf')
plt.close('all')

plot_muvs = [-22, -21, -20, -19, -18]
f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)

slopes = []
for muv in plot_muvs:
    left = muv - 0.5
    right = muv + 0.5
    leftidx = np.argmin(np.abs(abs_grid-left))
    rightidx = np.argmin(np.abs(abs_grid-right))
    y = np.log10(bestfit_z9)
    slope = y[rightidx] - y[leftidx]
    slopes.append(slope)

axs[0].plot(plot_muvs, slopes, 'k-',)

slopes = []
for muv in plot_muvs:
    left = muv - 0.5
    right = muv + 0.5
    leftidx = np.argmin(np.abs(abs_grid-left))
    rightidx = np.argmin(np.abs(abs_grid-right))
    y = np.log10(bestfit_z11)
    slope = y[rightidx] - y[leftidx]
    slopes.append(slope)

axs[1].plot(plot_muvs, slopes, 'k-',)
axs[0].set_title(r'$8.5<z<9.5$')
axs[1].set_title(r'$9.5<z<12.0$')
axs[0].set_xlim(-22.5, -17.5)
axs[1].set_xlim(-22.5, -17.5)
# axs[0].set_ylim(0, 1)
axs[0].set_ylabel(r'UVLF Slope $\frac{\mathrm{dLog}\phi_{\mathrm{UV}}}{\mathrm{d}M_{\mathrm{UV}}}$')
axs[0].set_xlabel(r'$M_{\mathrm{UV}}$')
axs[1].set_xlabel(r'$M_{\mathrm{UV}}$')
# axs[0].legend(frameon=False, fontsize=12.5)
# axs[1].legend(frameon=False, fontsize=12.5)
plt.savefig(f'uvlf_slope.pdf')
plt.close('all')