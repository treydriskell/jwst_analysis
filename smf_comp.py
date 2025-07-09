import h5py
import numpy as np
import os.path as path
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
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

    sortIdx = np.argsort(Mhs)
    Mhs = Mhs[sortIdx]
    stellarMass = stellarMass[sortIdx]
    treeWeights = treeWeights[sortIdx]
    data = np.stack((np.log10(Mhs), stellarMass),axis=-1)
    return data, treeWeights


def load_data(data_dir, z, reload=False):
    stellar_masses = []
    zs = []
    fn = data_dir+f'z{z}.hdf5'
    mhs_z = []
    mags_z = []
    weights_z = []
    f = h5py.File(fn,"r")

    d,w = get_Muv_from_hdf5(f)

    mhs = d[:,0]
    idx = np.argsort(mhs)
    mhs = mhs[idx]
    stellarMass = d[:,1][idx]
    w = w[idx]
    f.close()
    zs = np.array(zs)

    return mhs, zs, stellarMass, w  #abs_mag


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


    if z=='10.40':
        z10_mstar = [8.125, 8.875, 9.625]
        z10_phi = np.array([4.53e-4, 0.64e-4, 0.03e-4])
        z10_phi_upper = np.array([1.12e-4, 0.25e-4, 0.10e-4])
        z10_phi_lower = np.array([1.00e-4, 0.19e-4, 0.02e-4])

        lower = np.log10(z10_phi) - np.log10(z10_phi-z10_phi_lower)
        upper = np.log10(z10_phi+z10_phi_upper) - np.log10(z10_phi)

        ax.scatter(z10_mstar, np.log10(z10_phi),marker='o', s=20, facecolor=c, edgecolor=c, zorder=998)
        ax.errorbar(z10_mstar, np.log10(z10_phi), yerr=[lower,upper], marker='o', color="none", ls="none", ecolor=c, zorder=998, capsize=3.5)



def get_um_smf(z): 
    print(z)
    idx = (um_data['Z1'] < z) & (um_data['Z2'] > z)
    if np.sum(idx) <= 0:
        z_idx = np.argmin(np.abs(um_zs-z))
        z_close = um_zs[z_idx]
        idx = um_data['z'] == z_close
    log_sm = um_data['sm'][idx].to_numpy()
    logphi = um_data['Val'][idx].to_numpy()
    lower = um_data['Err_l'][idx].to_numpy()
    upper = um_data['Err_h'][idx].to_numpy()
    sort_idx = np.argsort(log_sm)
  
    return log_sm[sort_idx], logphi[sort_idx], upper[sort_idx], lower[sort_idx]


um_fn = 'data/obs.txt'
um_data = pd.read_csv(um_fn, sep='\s+', header=1) #, skiprows=1
um_data['z'] = (um_data['Z2'] + um_data['Z1'])/2.0
um_data['sm'] = (um_data['SM2'] + um_data['SM1'])/2.0
idx = (um_data['#Type'] == 'smf') & (um_data['Subtype'] == 'a')
um_data = um_data[idx]
um_zs = pd.unique(um_data['z'])

base_fn = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data/smf_params_p0/'

plot_zs = [0.1,0.7,1.0,2.,3.,4.,5.,6.,7.,8.]
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
 
plt.savefig('smf_comparison.pdf')

plt.close('all')

