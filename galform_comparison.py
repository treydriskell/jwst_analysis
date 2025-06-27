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
cosmo = FlatLambdaCDM(H0=67.36000, Om0=0.31530, Tcmb0=2.72548, Ob0=0.04930)
h = 0.7

def get_Muv_from_hdf5(outfile):
    outputs = outfile['Outputs']    
    nodeData = outputs['Output1']['nodeData']
    Mhs = nodeData['basicMass'][:]
    treeWeights = nodeData['mergerTreeWeight'][:]
    spheroidMassStellar = nodeData['spheroidMassStellar'][:]
    diskMassStellar = nodeData['diskMassStellar'][:]
    stellarMass = spheroidMassStellar+diskMassStellar

    z = nodeData['redshift'][0]
    sfn = 'spheroidLuminositiesStellar:JWST_NIRCAM_f277w:observed:z{:.4f}'.format(z)
    dfn = 'diskLuminositiesStellar:JWST_NIRCAM_f277w:observed:z{:.4f}'.format(z)
    Lum = nodeData[sfn][:]+nodeData[dfn][:] # changed
    Lum[Lum<1.0e-5] = 1.0e-5 
    abs_mag = -2.5*np.log10(Lum)

    sortIdx = np.argsort(Mhs)
    Mhs = Mhs[sortIdx]
    abs_mag = abs_mag[sortIdx]
    stellarMass = stellarMass[sortIdx]
    treeWeights = treeWeights[sortIdx]
    data = np.stack((np.log10(Mhs), stellarMass, abs_mag),axis=-1)
    return data, treeWeights


def load_data(data_dir, z, reload=False):
    # data_fn = path.join(data_dir, 'data.npy')
    stellar_masses = []
    # logmhs = []
    zs = []
    # bands = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    # for z in ['0.1']: #
    # for z in ['8.0']:
    print(z)
    # outfiles = [data_dir+f'z{z}:MPI{i:04d}.hdf5' for i in range(160)] #glob.glob(data_dir+f'z{z}:MPI*.hdf5')
    fn = data_dir+f'z{z}.hdf5'
    mhs_z = []
    mags_z = []
    weights_z = []
    print(fn)
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
    abs_mag = d[:,2][idx]
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
    return mhs, zs, stellarMass, abs_mag


# def behroozi_smhm_relation(logmh):
#     """ From https://arxiv.org/pdf/1806.07893 """
#     epsilon_0 = -1.431
#     epsilon_a = 1.757
#     epsilon_lna = 1.350
#     epsilon_z = -0.218
#     M_0 = 12.074
#     M_a = 4.600
#     alpha_lna = -1.816
#     alpha_z = 0.182
#     beta_0 = 0.470
#     beta_a = -0.875
#     beta_z - -0.487
#     delta_0 = 0.386
#     log10_M1 = M0 + Ma * (a-1) - Mlna * np.log(a) + Mz*z
#     epsilon = epsilon_0 + epsilon_a * (a-1) - epsilon_loga * np.log(a) + epsilon_z * z
#     alpha = alpha_0 + alpha_a * (a-1) - alpha_loga * np.log(a) + alpha_z * z
#     beta = beta_0 + beta_a * (a-1) + beta_z * z
#     log10_gamma = gamma_0 + gamma_a * (a-1) + gamma_z * z
#     gamma = 10**log10_gamma
#     x = logmh-log10_M1
#     log10_Mstar = epsilon - np.log10(10**(-alpha*x)+10**(-beta*x))+gamma*np.exp(-0.5*(x/delta)**2) + log10_M1

# gal_dirs = [
#             '/data001/gdriskell/jwst_blitz_astro_samples/nr4_p60/']    


# fn = '/data001/gdriskell/jwst_blitz_astro_samples/nr4_p60/'
fn = '/carnegie/nobackup/users/gdriskell/jwst_data/paper_params_p13845/'
plot_zs = ['8.0', '12.0', '16.0']

f, axs = plt.subplots(1,2, figsize=(12,5), constrained_layout=True) 
for i in range(3):
    # um_data = np.genfromtxt('uncertainties_smhm_true_med_cen.dat', names=True)
    # print(um_files[i])
    # print(um_data)
    # print(um_data[0])
    c = plt.cm.Dark2(i)

    # data_dir = '/data001/gdriskell/jwst_blitz_astro_samples/z0_params_p0/'
    logmhs, zs, stellar_masses, abs_mag = load_data(fn, plot_zs[i])
    # print(stellar_masses)
    # print(logmhs.shape, stellar_masses.shape)

    logmh_min = 9.0
    
    logmh_max = 11.5
    
    dlogmh = 0.2
    nbins = int(round((logmh_max-logmh_min)/dlogmh))
    bins = np.linspace(logmh_min, logmh_max, nbins)
    bin_centers = bins[:-1]+dlogmh/2.0
    Mstar_median = np.zeros_like(bin_centers)
    Mstar_scatter = np.zeros_like(bin_centers)
    median_smhm_ratio = np.zeros_like(bin_centers)
    abs_mag_median = np.zeros_like(bin_centers)
    abs_mag_scatter = np.zeros_like(bin_centers)

    for j in range(len(bin_centers)):
        left = bins[j]
        right = bins[j+1]
        idx = (logmhs > left) & (logmhs <= right)
        Mstar_median[j] = np.median(np.log10(stellar_masses[idx]))
        Mstar_scatter[j] = np.std(np.log10(stellar_masses[idx]))
        median_smhm_ratio[j] = np.median(np.log10(stellar_masses[idx])-logmhs[idx])
        abs_mag_median[j] = np.median(abs_mag[idx])
        abs_mag_scatter[j] = np.std(abs_mag[idx])

    # print(Mstar_average)
    # print(bin_centers.shape, median_smhm_ratio.shape)
    # plt.scatter(logmhs, np.log10(Mstar_average), marker = 'x', c='k', label='Galacticus data')
    # axs[0].plot(Mstar_median+np.log10(h), bin_centers+np.log10(h), label=f'$z={plot_zs[i]}$')
    axs[0].plot(bin_centers, Mstar_median, c=c, label=f'$z={plot_zs[i]}$')
    # axs[0].errorbar(bin_centers, Mstar_median, yerr=Mstar_scatter, c=c, label=f'$z={plot_zs[i]}$')
    if i==1:
        axs[0].fill_between(bin_centers, Mstar_median-Mstar_scatter, Mstar_median+Mstar_scatter, color=c, alpha=0.5)
    axs[0].set_xlabel(r'$\mathrm{Log}\,M_{h}\mathrm{[M_{\odot}]}$')
    axs[0].set_ylabel(r'Median $\mathrm{Log}\,M_{\ast}\mathrm{[M_{\odot}]}$')
    axs[0].set_ylim(6.5, 10.25)
    axs[0].set_xlim(9.0, 11.5)

    # axs[1].plot(abs_mag_median, bin_centers+np.log10(h), label=f'$z={plot_zs[i]}$')
    axs[1].plot(bin_centers, abs_mag_median, c=c, label=f'$z={plot_zs[i]}$')
    # axs[1].errorbar(bin_centers, abs_mag_median, yerr=abs_mag_scatter, c=c, label=f'$z={plot_zs[i]}$')
    axs[1].fill_between(bin_centers,  abs_mag_median-abs_mag_scatter, abs_mag_median+abs_mag_scatter, color=c, alpha=0.5)
    axs[1].set_xlabel(r'$\mathrm{Log}\,M_{h} \mathrm{[M_{\odot}]}$')
    axs[1].set_ylabel(r'Median $M_{\mathrm{uv}}$')
    axs[1].set_ylim(-15, -25)
    axs[1].set_xlim(9.0, 11.5)
    # axs[i].set_title(f'$z={plot_zs[i]}$')
axs[0].legend(frameon=False, fontsize=12, loc='upper left')
axs[1].legend(frameon=False, fontsize=12, loc='upper left')

plt.savefig('galaxy_halo_connection.pdf')
plt.close('all')

