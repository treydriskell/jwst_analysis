
import h5py
import numpy as np
# import os
import os.path as path
import analysis
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

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

# base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data'


f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)

df = pd.read_csv('test_lower_tau0.csv')
df.sort_values(by='sfr_timescale', inplace=True)

axs[0].plot(df['sfr_timescale'], df['like'])
axs[0].set_ylabel(r'$\mathcal{L}$')
axs[0].set_xlabel(r'$\tau_0$')

axs[1].plot(df['sfr_timescale'], df['loglike'])
axs[1].set_ylabel(r'$\mathrm{Log}(\mathcal{L})$')
axs[1].set_xlabel(r'$\tau_0$')

plt.savefig('lower_tau0_like.pdf')