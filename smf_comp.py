import h5py
import numpy as np
import os.path as path
import pickle as pkl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import itertools
# import yaml 
# import corner
# import pandas as pd
# from scipy.stats import chi2
import seaborn as sns
import glob
import pandas as pd

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['xtick.major.size'] = 6#10
mpl.rcParams['ytick.major.size'] = 6#10
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'

global cosmo
cosmo = FlatLambdaCDM(H0=68.000, Om0=0.307, Tcmb0=2.72548, Ob0=0.04850)

def get_Muv_from_hdf5(outfile):
    outputs = outfile['Outputs']    
    nodeData = outputs['Output1']['nodeData']
    Mhs = nodeData['basicMass'][:]
    treeWeights = nodeData['mergerTreeWeight'][:]
    spheroidMassStellar = nodeData['spheroidMassStellar'][:]
    diskMassStellar = nodeData['diskMassStellar'][:]
    stellarMass = spheroidMassStellar+diskMassStellar


    z = nodeData['redshift'][0]

    # app_mag = abs_mag+cosmo.distmod(z).value-2.5*np.log10(1+z)

    sortIdx = np.argsort(Mhs)
    Mhs = Mhs[sortIdx]
    stellarMass = stellarMass[sortIdx]
    treeWeights = treeWeights[sortIdx]
    data = np.stack((np.log10(Mhs), stellarMass),axis=-1)
    return data, treeWeights


def load_data(data_dir, z, reload=False):
    # data_fn = path.join(data_dir, 'data.npy')
    stellar_masses = []
    # logmhs = []
    zs = []
    # bands = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    # for z in ['0.1']: #
    # for z in ['8.0']:
    # print(z)
    # outfiles = [data_dir+f'z{z}:MPI{i:04d}.hdf5' for i in range(160)] #glob.glob(data_dir+f'z{z}:MPI*.hdf5')
    fn = data_dir+f'z{z}.hdf5'
    mhs_z = []
    mags_z = []
    weights_z = []
    # print(fn)
        # print(fn)
    f = h5py.File(fn,"r")
    # if 'Outputs' in f:
    # for band in bands:
    d,w = get_Muv_from_hdf5(f)
    # mhs_z.append(d[:,0])
    # mags_z.append(d[:,1])
    # weights_z.append(w)
    mhs = d[:,0]
    idx = np.argsort(mhs)
    mhs = mhs[idx]
    stellarMass = d[:,1][idx]
    # abs_mag = d[:,1][idx]
    w = w[idx]
    # save_uvlf(data_dir, mhs, mags, w, z, band)
    f.close()
    # print(stellarMass.shape)
            
        # raise Exception()
    # logmhs.append(mhs)
    # stellar_masses.append(stellarMass)
        # abs_mags.append(abs)
        # zs.extend([float(z) for i in range(len(logmhs[-1]))])
    
    # app_mags = np.concatenate(app_mags)
    # stellar_masses = np.concatenate(stellar_masses)
    # logmhs = np.concatenate(logmhs)
    zs = np.array(zs)
    # data = np.stack([logmhs, zs, abs_mags], axis=1)
    # np.save(data_fn, data)
    return mhs, zs, stellarMass, w  #abs_mag


# def schechter_function(logM, log_Mstar, phi1, alpha1, phi2, alpha2):
#     # phi_star = phi_star*1.0e-2
#     # log_Mstar = ...
#     term1 = phi1 * (10**(logM-log_Mstar))**(alpha1+1)
#     term2  = phi2 * (10**(logM-log_Mstar))**(alpha2+1)
#     return (np.log(10) * np.e**(-10**(logM-log_Mstar))
#             * (term1 + term2))


def add_jwst_data(ax, z, c):
    if z==8.:
        z8_mstar = [8.125, 8.875, 9.625]
        z8_phi = np.array([16.06e-4, 1.64e-4, 0.17e-4])
        z8_phi_upper = np.array([3.68e-4, 0.6e-4, 0.17e-4])
        z8_phi_lower = np.array([3.48e-4, 0.45e-4, 0.08e-4])
        lower = np.log10(z8_phi) - np.log10(z8_phi-z8_phi_lower)
        upper = np.log10(z8_phi+z8_phi_upper) - np.log10(z8_phi)

        ax.scatter(z8_mstar, np.log10(z8_phi), marker='o', s=20, facecolor=c, edgecolor=c, zorder=998)
        ax.errorbar(z8_mstar, np.log10(z8_phi), yerr=[lower,upper], marker='o', color="none", ls="none", ecolor=c, zorder=998, capsize=3.5)


    # z9_mstar = [8.125, 8.875, 9.625]
    # z9_phi = [8.16e-4, 0.5e-4, 0.05e-4]
    # z9_phi_upper = [2.3e-4, 0.35e-4, 0.10e-4]
    # z9_phi_lower = [2.13e-4, 0.2e-4, 0.05e-4]

    if z=='10.40':
        z10_mstar = [8.125, 8.875, 9.625]
        z10_phi = np.array([4.53e-4, 0.64e-4, 0.03e-4])
        z10_phi_upper = np.array([1.12e-4, 0.25e-4, 0.10e-4])
        z10_phi_lower = np.array([1.00e-4, 0.19e-4, 0.02e-4])

        lower = np.log10(z10_phi) - np.log10(z10_phi-z10_phi_lower)
        upper = np.log10(z10_phi+z10_phi_upper) - np.log10(z10_phi)

        ax.scatter(z10_mstar, np.log10(z10_phi),marker='o', s=20, facecolor=c, edgecolor=c, zorder=998)
        ax.errorbar(z10_mstar, np.log10(z10_phi), yerr=[lower,upper], marker='o', color="none", ls="none", ecolor=c, zorder=998, capsize=3.5)

    # z12_mstar = [8.125, 8.875]
    # z12_phi = [2.13e-4, 0.22e-4]
    # z12_phi_upper = [0.99e-4, 0.22e-4]
    # z12_phi_lower = [0.90e-4, 0.12e-4]



def get_um_smf(z):
    
    print(z)

    idx = (um_data['Z1'] < z) & (um_data['Z2'] > z)
    if np.sum(idx) <= 0:
        z_idx = np.argmin(np.abs(um_zs-z))
        z_close = um_zs[z_idx]
        idx = um_data['z'] == z_close
        #return None, None, None, None
    # print(um_data[idx])
    
    log_sm = um_data['sm'][idx].to_numpy()
    logphi = um_data['Val'][idx].to_numpy()
    lower = um_data['Err_l'][idx].to_numpy()
    upper = um_data['Err_h'][idx].to_numpy()
    # print(log_sm)
    sort_idx = np.argsort(log_sm)
    # print()
    # fn = um_fns[idx]
    # print(um_zs[idx])
    # log_sm, phi, upper_err, lower_err =  np.loadtxt(fn, usecols=[0,1,2,3]).T
    # print(lower_err[-1])
    # idx2 = lower_err < phi
    # logphi = np.log10(phi)
    # upper = np.log10(phi+upper_err)-logphi
    # lower = logphi-np.log10(phi-lower_err)
    return log_sm[sort_idx], logphi[sort_idx], upper[sort_idx], lower[sort_idx]
    # return log_sm[idx2], logphi[idx2], upper[idx2], lower[idx2]


# um_fns = glob.glob('umachine-dr1/data/obs.txt')
# um_zs = np.zeros_like(um_fns, dtype=float)
# for i,fn in enumerate(um_fns):
#     left = fn.rfind('/')
#     right = fn.rfind('.')
#     a = float(fn[left+6:right])
#     # if i==0:
#     # print(a)
#     z = 1./a - 1.
#     um_zs[i] = z
# print(um_zs.max(),um_zs.min())

um_fn = '~/umachine-dr1/data/obs.txt'
um_data = pd.read_csv(um_fn, sep='\s+', header=1) #, skiprows=1
um_data['z'] = (um_data['Z2'] + um_data['Z1'])/2.0
um_data['sm'] = (um_data['SM2'] + um_data['SM1'])/2.0
idx = (um_data['#Type'] == 'smf') & (um_data['Subtype'] == 'a')
um_data = um_data[idx]
um_zs = pd.unique(um_data['z'])
# print(pd.unique(um_data['Z1']),pd.unique(um_data['Z2']))
# print(um_zs)
# print()
# print(um_data.head())
# raise Exception()


# schechter_function = 

base_fn = '/carnegie/nobackup/users/gdriskell/jwst_data/smf_params_p0/'


# print
# plot_zs = ['0.1', '8.0', '12.0']
# plot_zs = ['0.35']
plot_zs = [0.1,0.7,1.0,2.,3.,4.,5.,6.,7.,8.]#, 8.86, '10.40', 11.94] # [0.1,0.7,1.0,2.,3.,4.,5.,6.,7.,8.]
nz = len(plot_zs)
f, ax = plt.subplots(1,1, figsize=(6,5), constrained_layout=True) 

for i in range(1,nz,2):
# for i in range(nz):
    z = plot_zs[i]
    # print(i,z)
    # ax = axs[i]
    # if (nz == 1):
    #     ax = axs
    # else:
    #     ax = axs[i]
    c = plt.cm.tab10(i//2)

    # data_dir = '/data001/gdriskell/jwst_blitz_astro_samples/z0_params_p0/'
    logmhs, zs, stellar_masses, weights = load_data(base_fn, z)
    log_sm = np.log10(stellar_masses)
    # print(np.amax(log_sm))
    # print(stellar_masses)
    # print(logmhs.shape, stellar_masses.shape)

    # logmh_min = 
    logMstar_min  = 8
    logMstar_max = 12
    # if i > 0:
    #     logmh_max = 11.4
    # else:
    # logmh_max = 14.375
    dlogMstar = 0.25
    nbins = int(round((logMstar_max-logMstar_min)/dlogMstar))+1
    bins = np.linspace(logMstar_min, logMstar_max, nbins)
    bin_centers = bins[:-1]+dlogMstar/2.0
    phi = np.zeros_like(bin_centers)
    
    # cosmos2020_lower = np.ones_like(bin_centers)*np.inf
    # cosmos2020_upper = np.zeros_like(bin_centers)
    # logMstars = [10.89-0.14, 10.89, 10.89+0.14]
    # alpha1s = [-1.42-0.06, -1.42, -1.42+0.05]
    # phi1s = [0.73e-3-0.27e-3, 0.73e-3, 0.73e-3+0.25e-3]
    # alpha2s = [-0.46-0.46, -0.46, -0.46+0.50]
    # phi2s = [1.09e-3-0.54e-3,1.09e-3,1.09e-3+0.50e-3]
    # pvalues = [logMstars, alpha1s, phi1s, alpha2s, phi2s]
    # for values in itertools.product(*pvalues):
    #     # print(values)

    #     sf = schechter_function(bin_centers, values[0], values[2], values[1], values[4], values[3]) 
    #     cosmos2020_lower = np.minimum(cosmos2020_lower, sf)
    #     cosmos2020_upper = np.maximum(cosmos2020_upper, sf)
        # print(bin_centers[5], cosmos2020_lower[5], cosmos2020_upper[5])

    # cosmos2020_sf_bf = schechter_function(bin_centers, 10.89, 0.73e-3 ,-1.42, 1.09e-3,-0.46) 
    # colors = sns.color_palette("Blues",n_colors=2)
    # c = colors[-1]

    for j in range(len(bin_centers)):
        left = bins[j]
        right = bins[j+1]
        idx = (log_sm > left) & (log_sm <= right)
        phi[j] = np.sum(weights[idx])
        
        # smhm_ratio_scatter[j] = np.std(np.log10(stellar_masses[idx])-logmhs[idx])
    phi = np.log10(phi/dlogMstar)
    ax.plot(bin_centers, phi, '-', c=c)
   
    # ax.fill_between(bin_centers, np.log10(cosmos2020_lower), np.log10(cosmos2020_upper), color=c, alpha=0.5, zorder=-1, 
    #                 label=f'Cosmos2020 $68\%$ CI')
    # ax.plot(bin_centers, np.log10(cosmos2020_sf_bf), c=c, label='COSMOS')
    # ax.plot(bin_centers, phi, c='k', label='This work')
    if float(z) < 8.0:
        log_sm, phi, upper, lower = get_um_smf(z)    
        ax.scatter(log_sm, phi, marker='o', s=20, facecolor=c, edgecolor=c, zorder=998)
        ax.errorbar(log_sm, phi, yerr=[lower,upper], marker='o', color="none", ls="none", ecolor=c, zorder=998, capsize=3.5)
    else:
        add_jwst_data(ax, z, c)
    if type(z)==float:
        ax.plot([],[],c=c, label=f'$z={z:.1f}$')
    else:
        ax.plot([],[],c=c, label=f'$z={z}$')

    # ax.plot(um_data[:,0], um_data[:,2], c='tab:orange', label='Median UniverseMachine')

ax.set_xlabel(r'$\mathrm{Log}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\mathrm{Log}(\Phi \mathrm{[Mpc^{-3} dex^{-1}]})$')
# ax.set_title(f'$z={plot_zs[i]}$')
ax.errorbar([], [], yerr=[], marker='o', color='k', ecolor='k', ls="none", capsize=3.5, label='Observations')
# ax.scatter([], [], marker='o', facecolor=c, edgecolor=c, label='UniverseMachine')
ax.plot([],[],'k-', label='This work')
ax.legend(frameon=False,fontsize=10)
ax.set_ylim(-7.5,-0.5)
ax.set_xlim(8,12.)
 
# plt.savefig('um_smf_data.pdf')
plt.savefig('smf_comparison.pdf')

plt.close('all')

