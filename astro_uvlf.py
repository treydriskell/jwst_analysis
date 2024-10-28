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


def parameters_to_labels(parameters):
    labels = []
    # for p in parameters:
    if p == 'velocityOutflow':
        # tex = r'$\mathrm{Log}(V_{\mathrm{outflow}})$'
        tex = r'$\left(V_{\mathrm{outflow}}\right)$'
    elif p == 'alphaOutflow':
        # tex = r'$\mathrm{Log}(\alpha_{\mathrm{outflow}})$'
        tex = r'$\left(\alpha_{\mathrm{outflow}}\right)$'
    elif p == 'starFormationFrequencyNormalization':
        # tex = r'$\mathrm{Log}(\nu_{0,\mathrm{SF}})$'
        tex = r'$\nu_{0,\mathrm{SF}}$'
    elif p == 'surfaceDensityExponent':
        # tex = r'$\mathrm{Log}(\alpha_{\mathrm{HI}})$'
        tex = r'$\alpha_{\mathrm{HI}}$'
    elif p == 'efficiency':
        tex = r'$\epsilon_{\star}$'
    elif p == 'timescale':
        tex = r'$\left(\tau_{0}\right)$'
    elif p == 'alphaStar':
        tex = r'$\left(\alpha_{\star}\right)$'
    else:
        raise Exception('Unknown Astro Parameter')
    return tex #labels

def title_labels(p):
    # tex = tex_labels(p)
    if p=='alphaOutflow':
        label = r'$\mathrm{Feedback\, Exponent}\, \alpha_{\mathrm{outflow}}$'
    if p=='alphaStar':
        label = r'$\mathrm{SFE\,Exponent}\, \alpha_{\mathrm{\star}}$'
    if p=='timescale':
        label = r'$\mathrm{SFE\,Norm.}\,\tau_{0}$'
    if p=='velocityOutflow':
        label = r'$\mathrm{Feedback\,Norm.}\,V_{\mathrm{outflow}}$'
    return label

def tex_labels(p):
    if p=='alphaOutflow':
        label = r'$\alpha_{\mathrm{outflow}}$'
    if p=='alphaStar':
        label = r'$\alpha_{\mathrm{\star}}$'
    if p=='timescale':
        label = r'$\tau_{0}$'
    if p=='velocityOutflow':
        label = r'$V_{\mathrm{outflow}}$'
    return label

# def get_idx(a,b,c,d):
#     return a+3*b+9*c+27*d

# print(2,0,0,0)
# a = alpha_star
# b = tau
# c = alpha_outflow
# d = vOutflow

base_dir = '/data001/gdriskell/jwst_blitz_astro_samples/'
df = pd.read_csv('final_df.csv')

cmaps = ['ember', 'amythest', 'freeze', 'nuclear']

params = ['alphaStar','timescale','alphaOutflow','velocityOutflow']

global nz
nz = 17
global z_grid
z_grid = np.linspace(8.0, 16.0, nz)
dz = z_grid[1]-z_grid[0]

abs_min = -25.0
abs_max = 0.
dabs = 0.2
nabs = int(round((abs_max-abs_min)/dabs))+1
abs_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = abs_grid[1]-abs_grid[0]

dMh = 0.15
Mh_bins = np.arange(8.0, 11.76, dMh)
global nm
nm = len(Mh_bins)-1
bin_centers = Mh_bins[:-1] + dMh/2.0 

zidx = z_grid==8.0


bestfit_alphaStar = -1.5
bestfit_timescale = 0.1
bestfit_velocityOutflow = 150
bestfit_alphaOutflow = 1.5

for p in params:
    # print(p)
    f = plt.figure()
    colors = iter([plt.cm.Dark2(i) for i in range(8)])
    title = title_labels(p)
    tex = tex_labels(p)
    if p=='alphaStar':
        cmap = 'cmr.ember'
        param_path = 'starFormationRateDisks/starFormationTimescale/exponentVelocity'
        # idx1 = get_idx(0,1,2,0)
        # idx2 = get_idx(1,1,2,0)
        # idx3 = get_idx(2,1,2,0)
        # values = [-3, -2, -1]

        values = [-4.0, -2.5, -1.0]
    elif p=='timescale':
        cmap = 'cmr.bubblegum'
        param_path = 'starFormationRateDisks/starFormationTimescale/timescale'
        # idx1 = get_idx(0,1,0,0)
        # idx2 = get_idx(1,1,0,0)
        # idx3 = get_idx(2,1,0,0)
        values = [0.1, 0.5, 0.9]
    elif p=='alphaOutflow':
        cmap = 'cmr.sapphire'
        param_path = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
        # idx1 = get_idx(1,1,0,1)
        # idx2 = get_idx(1,1,1,1)
        # idx3 = get_idx(1,1,2,1)
        values = [0.5, 1.5, 2.5]
    elif p=='velocityOutflow':
        cmap = 'cmr.nuclear'
        param_path = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
        # idx1 = get_idx(0,0,0,0)
        # idx2 = get_idx(1,0,0,0)
        # idx3 = get_idx(2,0,0,0)
        values = [100, 150, 200]
    for v in values:
        # print(v)
        label=tex+f'$={v}$'
        # print(df['alphaStar'].unique())
        # print(p,idx)
        cmap_sub = cmr.get_sub_cmap(cmap, 0.25, 0.75)
        norm = mpl.colors.Normalize(vmin=np.amin(values), vmax=np.amax(values))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_sub)
        # if (p=='alphaStar') or p=='alphaOutflow':
        #     data_dir = f'/data001/gdriskell/jwst_blitz_astro_samples/fixed_ngdeep_params_p{idx}/'
        # elif p=='timescale':
        #     data_dir = f'/data001/gdriskell/jwst_blitz_astro_samples/plot_params_p{idx}/'
        # elif p=='velocityOutflow':
        #     data_dir = f'/data001/gdriskell/jwst_blitz_astro_samples/plot_params2_p{idx}/'
        if p=='timescale':
            idx = (
                (df['alphaOutflow']==bestfit_alphaOutflow) & 
                (df['velocityOutflow']==bestfit_velocityOutflow) & 
                (df['timescale']== v) &
                (df['alphaStar']==bestfit_alphaStar)
            )
            label += r'$\,\mathrm{Gyr}$'
        elif p=='alphaOutflow':
            idx = (
                (df['alphaOutflow']==v) & 
                (df['velocityOutflow']==bestfit_velocityOutflow) & 
                (df['timescale']== bestfit_timescale) &
                (df['alphaStar']==bestfit_alphaStar)
            )
        elif p=='alphaStar':
            idx = (
                (df['alphaOutflow']==bestfit_alphaOutflow) & 
                (df['velocityOutflow']==bestfit_velocityOutflow) & 
                (df['timescale']== bestfit_timescale) &
                (df['alphaStar']==v)
            )
        elif p=='velocityOutflow':
            idx = (
                (df['alphaOutflow']==bestfit_alphaOutflow) & 
                (df['velocityOutflow']==v) & 
                (df['timescale']== bestfit_timescale) &
                (df['alphaStar']==bestfit_alphaStar)
            )
            label += r'$\,\mathrm{km/s}$'
            
        row = df[idx]
        # print(row['tag'].values[0])
        tag = row['tag'].values[0]
        pidx = row['idx'].values[0]
        data_dir = path.join(base_dir, f'{tag}_p{pidx}/')

        uvlf_fn = path.join(data_dir, 'interp_uvlf_abs.npy')
        uvlf = np.load(uvlf_fn)
        uvlf = uvlf[:,zidx]/dabs
        # color = next(colors)
        
        plt.plot(abs_grid, np.log10(uvlf), lw=2.5, color=sm.to_rgba(v), label=label) # label='JWST best fit'
    # plt.title(parameters_to_labels(p))
    # plt.title(title)
    ax = plt.gca()
    # ax.axes.yaxis.set_ticklabels([])
    plt.xlabel(r'$M_{\mathrm{UV}}$')
    plt.ylabel(r'$\mathrm{Log}(\phi_{\mathrm{UV}}\,/\,\mathrm{Mpc^{-3} mag^{-1}})$')
    plt.xlim(-23, -16.5)
    # plt.ylim()
    plt.ylim(-6.5, -1.0)
    plt.legend(frameon=False, fontsize=13.5)
    plt.tight_layout()
    plt.savefig(f'uvlf_{p}.pdf')
    plt.close('all')

# [bowler2022_z8_lower_err, bowler2022_z8_upper_err]

