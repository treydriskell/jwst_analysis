
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
import analysis

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
cosmo = FlatLambdaCDM(H0=67.36000, Om0=0.31530, Tcmb0=2.72548, Ob0=0.04930)


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


def plot_uvlf(axs, muv_grid, uvlf, data_dir, abs=False):
    c = 'k' #s_m.to_rgba(i)
    # c = plt.cm.Dark2(3)
    # zs = [8.0, 12.0, 16.0]
    # for i,z in enumerate(zs):
    ax = axs[0]
    zidx = (analysis.redshift_grid > 8.5) & (analysis.redshift_grid < 9.5)
    # if abs:
    #     analysis.absolute_magnitude_grid = muv_grid
    # else:
    #     analysis.absolute_magnitude_grid = muv_grid - cosmo.distmod(9.0).value+2.5*np.log10(1+9.0)
    #     dm = (analysis.absolute_magnitude_grid[2:]-analysis.absolute_magnitude_grid[:-2])/2.0
    # print(np.sum(zidx))
    y = np.average(uvlf[:, zidx],axis=1) 
    # y = uvlf[:, zidx][:,-1]
    # print(np.amin(y), np.amax(y))
    midx = y>0
    y = np.log10(y[midx]) # /dm[midx]
    ax.plot(analysis.absolute_magnitude_grid[midx], y, lw=2.0, c=c, label='Best fit') # label='Sim.',
    ax = axs[1]
    zidx = (analysis.redshift_grid > 9.5) & (analysis.redshift_grid < 12.0)
    # if abs:
    #     analysis.absolute_magnitude_grid = muv_grid
    #     dm = analysis.absolute_magnitude_grid[1]-analysis.absolute_magnitude_grid[0]
    # else:
    #     analysis.absolute_magnitude_grid = muv_grid - cosmo.distmod(10.5).value+2.5*np.log10(1+10.5)
    #     # dm = np.array([analysis.absolute_magnitude_grid[1]-analysis.absolute_magnitude_grid[0], analysis.absolute_magnitude_grid[1:]-analysis.absolute_magnitude_grid[:-1], analysis.absolute_magnitude_grid[-1]-analysis.absolute_magnitude_grid[-2]])
    #     dm = ((analysis.absolute_magnitude_grid[1:-1]-analysis.absolute_magnitude_grid[:-2]) + (analysis.absolute_magnitude_grid[2:]-analysis.absolute_magnitude_grid[1:-1]))/2.0
    y = np.average(uvlf[:, zidx],axis=1)#[1:-1]
    midx = y>0
    y = np.log10(y[midx]) # /dm[midx]
    ax.plot(analysis.absolute_magnitude_grid[midx], y, lw=2.0, c=c, label='Best fit') #  label='Sim.',
    

def plot_data(ax):
    Bouwens2021_Muv = [-21.85, -21.35, -20.85, -20.10, -19.35, -18.60, -17.60]
    Bouwens2021_phi = np.array([0.000003, 0.000012, 0.000041, 0.000120, 0.000657, 0.001100, 0.003020])
    Bouwens2021_err = np.array([0.000002, 0.000004, 0.000011, 0.000040, 0.000233, 0.000340, 0.001140])
    Bouwens2021_log_phi = np.log10(Bouwens2021_phi)
    Bouwens2021_log_err = Bouwens2021_log_phi-np.log10(Bouwens2021_phi-Bouwens2021_err)

    Bowler2020_Muv = [-21.65, -22.15, -22.90]
    Bowler2020_phi = np.array([2.95e-6, 0.58e-6, 0.14e-6])
    Bowler2020_err = np.array([0.98e-6, 0.33e-6, 0.06e-6])
    Bowler2020_log_phi = np.log10(Bowler2020_phi)
    Bowler2020_log_err = Bowler2020_log_phi-np.log10(Bowler2020_phi-Bowler2020_err)

    Stefanon2019_Muv = [-22.55, -22.05, -21.55]
    Stefanon2019_phi = np.array([0.76e-6, 1.38e-6, 4.87e-6])
    Stefanon2019_err_upper = np.array([0.74e-6, 1.09e-6, 2.01e-6])
    Stefanon2019_err_lower = np.array([0.41e-6, 0.66e-6, 1.41e-6])
    Stefanon2019_log_phi = np.log10(Stefanon2019_phi)
    Stefanon2019_log_err = [Stefanon2019_log_phi-np.log10(Stefanon2019_phi-Stefanon2019_err_lower),
                            np.log10(Stefanon2019_phi+Stefanon2019_err_upper)-Stefanon2019_log_phi]

    Mclure2013_Muv = [-21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.25]
    Mclure2013_phi = np.array([0.000008, 0.00003, 0.0001, 0.0003, 0.0005, 0.0012, 0.0018, 0.0028, 0.0050])
    Mclure2013_err = np.array([0.000003, 0.000009, 0.00003, 0.00006, 0.00012, 0.0004, 0.0006, 0.0008, 0.0025])
    Mclure2013_log_phi = np.log10(Mclure2013_phi)
    Mclure2013_log_err = Mclure2013_log_phi-np.log10(Mclure2013_phi-Mclure2013_err)

    c = plt.cm.Dark2(0)
    ax.scatter(Bouwens2021_Muv, Bouwens2021_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor=c,label='Bouwens 2021')
    ax.errorbar(Bouwens2021_Muv, Bouwens2021_log_phi, yerr=Bouwens2021_log_err, elinewidth=1.8, marker='o', color="none", ecolor=c, ls='none', capsize=3.5)

    c = plt.cm.Dark2(1)
    ax.scatter(Bowler2020_Muv, Bowler2020_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor=c,label='Bowler 2020')
    ax.errorbar(Bowler2020_Muv, Bowler2020_log_phi, yerr=Bowler2020_log_err, elinewidth=1.8, marker='o', color="none", ecolor=c, ls='none', capsize=3.5)

    c = plt.cm.Dark2(2)
    ax.scatter(Stefanon2019_Muv, Stefanon2019_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor=c,label='Stefanon 2019')
    ax.errorbar(Stefanon2019_Muv, Stefanon2019_log_phi, yerr=Stefanon2019_log_err, elinewidth=1.8, marker='o', color="none", ecolor=c, ls='none', capsize=3.5)

    c = plt.cm.Dark2(3)
    ax.scatter(Mclure2013_Muv, Mclure2013_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor=c,label='Mclure 2013')
    ax.errorbar(Mclure2013_Muv, Mclure2013_log_phi, yerr=Mclure2013_log_err, elinewidth=1.8, marker='o', color="none", ecolor=c, ls='none', capsize=3.5)

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


app_cutoff = 30.4

plot_data(plt.gca())


base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data/'
df = pd.read_csv('paper_params.csv')
df = df.sort_values('loglike', ascending=False)
df.insert(len(df.columns), 'like', np.exp(df['loglike']))

z_left = analysis.redshift_grid-analysis.dz/2.0
z_right = analysis.redshift_grid+analysis.dz/2.0

bffn = base_dir+f'paper_params_p13845/'
# print(fn)
uvlf = analysis.get_uvlf(None,None,bffn,True,False,False)/analysis.dabs
zidx = analysis.redshift_grid == 8.0
zs = analysis.redshift_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = analysis.z_volumes[zidx]
totalV = np.zeros_like(analysis.absolute_magnitude_grid)
y = np.zeros_like(analysis.absolute_magnitude_grid)
for j,muv in enumerate(analysis.absolute_magnitude_grid):
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
bestfit_z8 = y
plt.plot(analysis.absolute_magnitude_grid, np.log10(y), '-k', label='Best fit')


n_sample = 5000
# fns = [f'/data001/gdriskell/jwst_blitz_astro_samples/final_corners_p{i}' for i in ns]
colors = sns.color_palette("Blues",n_colors=2) 

# for i, n in enumerate(n_samples):
z8_upper = np.zeros_like(analysis.absolute_magnitude_grid)
z8_lower = np.ones_like(analysis.absolute_magnitude_grid)

# c = colors[-(i+1)]
# c = colors[-(i+1)]
samples = df.sample(n_sample, replace=True, weights='like')
z8_ys = []
for index,row in samples.iterrows():
    fn = base_dir+f'paper_params_p{index}/'
    if fn != bffn:
        # print(fn)
        uvlf = analysis.get_uvlf(None,None,fn,True,False,False)/analysis.dabs
        zidx = (analysis.redshift_grid == 8.0) 
        zs = analysis.redshift_grid[zidx]
        z_left_slice = z_left[zidx]
        z_right_slice = z_right[zidx]
        zuvlf = uvlf[:,zidx]
        zvolume =analysis.z_volumes[zidx]
        totalV = np.zeros_like(analysis.absolute_magnitude_grid)
        y = np.zeros_like(analysis.absolute_magnitude_grid)
        for j,muv in enumerate(analysis.absolute_magnitude_grid):
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
        z8_ys.append(y)


z8_ys = np.array(z8_ys)
z8_diff = np.abs(z8_ys-bestfit_z8) 

for i,frac in enumerate([0.68, 0.95]):
    z8_lower = np.zeros_like(analysis.absolute_magnitude_grid)
    z8_upper = np.zeros_like(analysis.absolute_magnitude_grid)
    c = colors[-(i+1)]
    n_cut = int(round(frac*n_sample))
    for im, m in enumerate(analysis.absolute_magnitude_grid):
        z8_idx = np.argsort(z8_diff[:,im])
        z8_diff_m = z8_diff[z8_idx,im]
        z8_ys_m = z8_ys[z8_idx,im]
        cut_uvlfs_z8 = z8_ys_m[:n_cut]
        z8_upper[im] = np.amax(cut_uvlfs_z8, axis=0)
        z8_lower[im] = np.amin(cut_uvlfs_z8, axis=0)


    idx = z8_lower > 0
    z8_upper = np.log10(z8_upper[idx])
    z8_lower = np.log10(z8_lower[idx])
    # ax = axs[0]
    # print(idx.shape)
    plt.fill_between(analysis.absolute_magnitude_grid[idx], z8_lower, z8_upper, color=c, alpha=0.9-i*0.2, zorder=-i, 
                    label=f'${int(round(frac*100))}'+r'\% $ CI')


plt.title(r'$z=8$')
plt.xlabel(r'$M_{\mathrm{UV}}$')
plt.xlim(-23, -17.0)
plt.ylim(-7.5, -1.5)
plt.ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()

plt.savefig(f'bestfit_hst_comparison.pdf')
plt.close('all')