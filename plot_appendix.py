
import h5py
import numpy as np
# import os
import os.path as path
from astropy.cosmology import FlatLambdaCDM
import itertools
import analysis
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

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

rng = np.random.default_rng()

# global cosmo
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)

base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data'

bestfit_directory = path.join(base_dir, 'paper_params_p13845/')
bestfit_sfr_alpha = -4.0
bestfit_sfr_timescale = 0.1
bestfit_outflow_velocity = 130
bestfit_outflow_alpha = 1.75

def get_data(hdf5_fn):
    f = h5py.File(hdf5_fn,"r")
    disk_velocity = f['Outputs']['Output1']['nodeData']['diskVelocity'][:]
    gas_mass = f['Outputs']['Output1']['nodeData']['diskMassGas'][:]
    halo_mass = f['Outputs']['Output1']['nodeData']['basicMass'][:]
    idx = np.argsort(halo_mass)
    halo_mass = halo_mass[idx]
    disk_velocity = disk_velocity[idx]
    gas_mass = gas_mass[idx]
    f.close()
    return halo_mass, disk_velocity, gas_mass


def calculate_SF_timescale(disk_velocity, sfr_timescale, sfr_alpha):
    return sfr_timescale * (disk_velocity / 50.0)**sfr_alpha

def calculate_SF_rate(gas_mass, disk_velocity, sfr_timescale, sfr_alpha):
    timescale = calculate_SF_timescale(disk_velocity, sfr_timescale, sfr_alpha)
    return gas_mass / timescale

def calculate_Moutflow_rate(gas_mass, disk_velocity, outflow_velocity, outflow_alpha, sfr_timescale, sfr_alpha):
    sfr_rate = calculate_SF_rate(gas_mass, disk_velocity, sfr_timescale, sfr_alpha)
    return (outflow_velocity / disk_velocity)**outflow_alpha * sfr_rate

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)

i = 0
for z in ['8.0', '12.0', '16.0'][::-1]:
    n = 50
    hdf5_fn = path.join(bestfit_directory, f'z{z}.hdf5')
    halo_mass, disk_velocity, gas_mass = get_data(hdf5_fn)
    idx = disk_velocity > 0
    halo_mass = halo_mass[idx]
    disk_velocity = disk_velocity[idx]
    gas_mass = gas_mass[idx]

    timescale = calculate_SF_timescale(disk_velocity, bestfit_sfr_timescale, bestfit_sfr_alpha)
    outflow_rate = calculate_Moutflow_rate(gas_mass, disk_velocity, bestfit_outflow_velocity, 
                                           bestfit_outflow_alpha, bestfit_sfr_timescale, bestfit_sfr_alpha)
    c = plt.cm.Dark2(i)
    z = int(float(z))

    average_outflow_rate = moving_average(outflow_rate,n=n)
    average_halo_mass = moving_average(halo_mass, n=n)
    axs[0].loglog(average_halo_mass, average_outflow_rate, color = c, label = f'$z={z}$')
    

    average_timescale = moving_average(timescale,n=n)
    axs[1].loglog(average_halo_mass, average_timescale, color = c, label = f'$z={z}$')
    i += 1

axs[0].set_ylabel(r'$\dot{M}_{\mathrm{outflow}}\,\mathrm{[M_{\odot}/Gyr]}$')
axs[0].set_xlabel(r'$M_{h}$')
axs[0].legend(frameon=False)

axs[1].set_ylabel(r'$\tau_{\star}\,\mathrm{[Gyr]}$')
axs[1].set_xlabel(r'$M_{h}$')
axs[1].legend(frameon=False)

plt.savefig('bestfit_outflow_timescale.pdf')