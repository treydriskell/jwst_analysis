
import h5py
import numpy as np
import scipy
import os
import os.path as path
import pickle as pkl
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM
import itertools
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import seaborn as sns
from scipy.special import gamma

rng = np.random.default_rng()

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['xtick.major.size'] = 5#10
mpl.rcParams['ytick.major.size'] = 5#10
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'

# global cosmo
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)


def get_weights_from_hmf(hmf_fn):
    """ Output1: z=17
    Output51: z=7
    """
    weights_fn = 'data/gpr_zgrid_weights.npy'
    if path.isfile(weights_fn):
        weights = np.load(weights_fn) 
    else:
        weights = []
        f = h5py.File(hmf_fn,"r")
        Mhs = np.geomspace(1.0e8,5.0e11,3699)
        logMhs = np.log10(Mhs)
        logdelta = logMhs[1]-logMhs[0]
        Mmin = 10**(logMhs-logdelta/2.0)
        Mmax = 10**(logMhs+logdelta/2.0)
        deltaM = Mmax-Mmin
        weights = []
        for i in range(17):
            j = 17-i
            output = f[f'Outputs/Output{j}']
            haloMass = output['haloMass'][:]
            hmf = output['haloMassFunctionM'] 
            weight = deltaM*hmf
            weights.append(weight)
        weights = np.array(weights)
        f.close()
        np.save(weights_fn,weights)
    return weights


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


def get_ngdeep_cf(app_grid):
    """z~9 covers z=8.5-9.5 and z~11 covers z=9.5-12.0
    Can't find survey area so just dividing by max for now"""
    z9_volume = cosmo.comoving_volume(9.5)-cosmo.comoving_volume(8.5)
    z11_volume = cosmo.comoving_volume(12.0)-cosmo.comoving_volume(9.5)
    ngdeep_absmag_z9 = np.array([-21.1, -20.1, -19.1, -18.35, -17.85, -17.35])
    ngdeep_appmag_z9 = ngdeep_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
    ngdeep_veff_z9 = np.array([18700., 18500., 15800., 13100., 7770., 2520.]) # Mpc^3
    # ngdeep_absmag_z11 = np.array([-20.05, -19.35, -18.65, -17.95, -17.25, -17.05]) # last value using extrapolation
    # ngdeep_veff_z11 = np.array([27900., 26100., 20800., 9840., 2210., 0.]) 
  
    # ngdeep_appmag_z11 = ngdeep_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
    ngdeep_z9_cf = ngdeep_veff_z9 / np.amax(ngdeep_veff_z9) #z9_volume
    # ngdeep_z11_cf = ngdeep_veff_z11 / np.amax(ngdeep_veff_z11) #z11_volume

    ### TODO: really dumb version for now <------!
    # print(ngdeep_appmag_z9)
    # print(ngdeep_appmag_z11)
    # ngd_cf = np.concatenate(([1.0], ngdeep_z11_cf, [0.0]))
    # ngd_app = np.concatenate(([np.amin(app_grid)], ngdeep_appmag_z11, [np.amax(app_grid)]))
    ngdeep_appmag_z9 = np.append(ngdeep_appmag_z9, 30.4)
    ngdeep_z9_cf = np.append(ngdeep_z9_cf, 0.0)
    
    ngd_cf = np.concatenate(([1.0], ngdeep_z9_cf, [0.0]))
    ngd_app = np.concatenate(([np.amin(app_grid)], ngdeep_appmag_z9, [np.amax(app_grid)]))
    cf = np.interp(app_grid, ngd_app, ngd_cf)
    return cf


def get_ceers_cf(app_grid):
    """z~9 covers z=8.5-9.7 and z~11 covers z=9.7-13.0, z~14 covers > 13
    Can't find survey area so just dividing by max for now"""
    z9_volume = cosmo.comoving_volume(9.7)-cosmo.comoving_volume(8.5)
    # z11_volume = cosmo.comoving_volume(13.0)-cosmo.comoving_volume(9.7)
    # z14_volume = cosmo.comoving_volume(14.5)-cosmo.comoving_volume(13.0)
    ceers_absmag_z9 = np.array([-22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5])
    ceers_appmag_z9 = ceers_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
    ceers_veff_z9 = np.array([187000., 187000., 187000., 193000., 177000., 161000., 120000., 77900., 18600.]) # Mpc^3
    
    # ceers_absmag_z11 = np.array([-21.0, -20.5, -20.0, -19.5, -19.0, -18.5, -18.0]) # last value using extrapolation
    # ceers_veff_z11 = np.array([373000., 354000., 306000., 220000., 62100., 23800., 0.0]) 
    # ceers_appmag_z11 = ceers_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
    
    # ceers_absmag_z14 = np.array([-20.5, -20.0, -19.5, -19.0,]) # last value using extrapolation
    # ceers_veff_z14 = np.array([147000., 85400., 60600., 0.0]) 
    # ceers_appmag_z14 = ceers_absmag_z11+cosmo.distmod(14.0).value-2.5*np.log10(1+14.0)

    ceers_z9_cf = ceers_veff_z9 / np.amax(ceers_veff_z9) #z9_volume
    # ceers_z11_cf = ceers_veff_z11 / np.amax(ceers_veff_z11) #z11_volume
    # ceers_z14_cf = ceers_veff_z14 / np.amax(ceers_veff_z14)

    ceers_appmag_z9 = np.append(ceers_appmag_z9, 29.15)
    ceers_z9_cf = np.append(ceers_z9_cf, 0.0)
    # print(ceers_appmag_z9)
    ceers_cf = np.concatenate(([1.0], ceers_z9_cf, [0.0]))
    ceers_app = np.concatenate(([np.amin(app_grid)], ceers_appmag_z9, [np.amax(app_grid)]))
    cf = np.interp(app_grid, ceers_app, ceers_cf)
    # print(cf)
    return cf


def get_data_pdf(observed_data, muv_grid, z_grid, pdf_fn, overwrite=False):
    # obs_pdf_fn = '/home/gdriskell/galacticus/data/ngdeep_pdf.npy'
    z_cutoff = 8.5 
    if path.isfile(pdf_fn) and not overwrite:
        obs_pdf = np.load(pdf_fn)
    else:
        ndata = len(observed_data)
        ndm = len(muv_grid)
        ndz = len(z_grid)
        obs_pdf = np.zeros((ndata,ndm,ndz))
        for i in range(ndata):
            muv = observed_data['mf277w'][i]
            z = observed_data['z'][i]
            z_upper_err = observed_data['z_upper_err'][i]
            z_lower_err = np.abs(observed_data['z_lower_err'][i])
            dz = z_grid[1]-z_grid[0]
            muv_prob = np.zeros_like(muv_grid)
            idx = np.argmin(np.abs(muv_grid-muv))
            # print(idx)
            muv_prob[idx] = 1.0
            muv_prob = muv_prob.reshape(-1, 1)
            lower_idx = z_grid <= z # note, choice of <= is arbitrary!
            upper_idx = z_grid > z
            z_pdf = np.zeros_like(z_grid)
            norm = np.sqrt(2.0/np.pi)/(z_upper_err + z_lower_err)
            z_pdf[lower_idx] = norm * np.exp(-(z_grid[lower_idx]-z)**2 / 2.0 / z_lower_err**2)
            z_pdf[upper_idx] = norm * np.exp(-(z_grid[upper_idx]-z)**2 / 2.0 / z_upper_err**2)
            z_pdf = z_pdf.reshape(1, -1)
            obs_pdf[i,:,:] = muv_prob * z_pdf * dz 
            cut_idx = z_grid < 8.5
            obs_pdf[i][:,cut_idx] = 0
            new_norm = np.sum(obs_pdf[i,:,:])
            obs_pdf[i,:,:] = obs_pdf[i,:,:] / new_norm
        np.save(pdf_fn, obs_pdf)
    return obs_pdf

    
def load_data(data_dir, reload):
    data_fn = path.join(data_dir, 'data.npy')
    if path.isfile(data_fn) and not reload:
        weights, logmhs, abs_mags, app_mags, zs = np.load(data_fn).T
    else:
        app_mags = []
        abs_mags = []
        logmhs = []
        zs = []
        weights = []
        for z in ['8.0', '12.0', '16.0']: #
            # print(z)
            #outfiles = [data_dir+f'z{z}:MPI{i:04d}.hdf5' for i in range(160)] #glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            # outfiles = glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            # outfiles = glob.glob(data_dir+f'z{z}.hdf5')
            # outfiles = glob.glob(data_dir+f'z{z}:MPI0000.hdf5')
            # outfiles += glob.glob(data_dir+f'lowMh_z{z}:MPI0000.hdf5')
            mhs = []
            app = []
            abs = []
            ws = []
            fn = data_dir+f'z{z}.hdf5'
            # for fn in outfiles:
            try: 
                f = h5py.File(fn,"r")
                if 'Outputs' in f:
                    d,w = get_Muv_from_hdf5(f, 'JWST_NIRCAM_f277w')
                    mhs.append(d[:,0].flatten())
                    app.append(d[:,1].flatten())
                    abs.append(d[:,2].flatten())
                    ws.append(w.flatten())
                    # print(len(mhs[-1]))
                f.close()
            except:
                print(f'issue with {fn}')
                continue
            mhs = np.concatenate(mhs)
            idx = np.argsort(mhs)
            mhs = mhs[idx]
            app = np.concatenate(app)[idx]
            abs = np.concatenate(abs)[idx]
            logmhs.append(mhs)
            app_mags.append(app)
            abs_mags.append(abs)
            weights.append(np.concatenate(ws)[idx])
            # zs.extend([float(z)] * len(mhs))
            zs.extend([float(z) for i in range(len(logmhs[-1]))])
        
        app_mags = np.concatenate(app_mags)
        weights = np.concatenate(weights)
        abs_mags = np.concatenate(abs_mags)
        logmhs = np.concatenate(logmhs)
        zs = np.array(zs)
        data = np.stack([weights, logmhs, abs_mags, app_mags, zs], axis=1)
        np.save(data_fn, data)
        # print(len(np.unique(logmhs)))
    return weights, logmhs, abs_mags, app_mags, zs


def get_binned_weights(logmh_bins, logmhs, weights, recompute=False):
    bweights_fn =  'data/binned_zgrid_weights.npy'
    if path.isfile(bweights_fn) and not recompute:
        bweights = np.load(bweights_fn)
    else:
        logmhs = np.unique(logmhs)
        bweights = np.zeros((nz,nm))
        for i in range(nm):
            left = Mh_bins[i]
            right = Mh_bins[i+1]
            bidx = (logmhs > left) & (logmhs < right)
            bweights[:,i] = np.sum(weights[:,bidx],axis=1)
        np.save(bweights_fn, bweights)
    return bweights


def get_probs(mag_grid, mean, std, mins, maxs, data_dir, output_tag, do_abs, recompute, save=True):
    """Returns array of probabilities evaluated on a grid of N_mag x N_z x N_mh"""
    if do_abs:
        probs_fn = path.join(data_dir, output_tag + '_probs_abs.npy')
    else:
        probs_fn = path.join(data_dir, output_tag + '_probs.npy')
    if path.isfile(probs_fn) and not recompute:
        probs = np.load(probs_fn)
    else:
        mag_grid = mag_grid.reshape(-1,1,1)
        mean = mean.reshape(1, nz, nm)
        std = std.reshape(1, nz, nm)
        mag_dim = len(mag_grid)
        probs = np.exp(-(mag_grid-mean)**2/2.0/std**2)/np.sqrt(2*np.pi)/std
        
        for i, mag in enumerate(mag_grid):
            if do_abs:
                idx = mag > (maxs * 0.9)
            else:
                idx = mag > (maxs * 1.05)
            probs[i][idx] = 0
            if do_abs:
                idx = mag < (mins * 1.1)
            else:
                idx = mag < (mins * 0.95)
            probs[i][idx] = 0

        norm = np.sum(probs, axis=0) 
        norm = norm.reshape(1, nz, nm)
        # np.isclose(norm, )
        # print(norm <= 0)
        
        probs = (probs / norm)
        if save:
            np.save(probs_fn, probs)
    return probs


def get_muvz_probs(mag_grid, mean, std, mins, maxs, bweights, data_dir, output_tag, do_abs, recompute, save=True):
    """Returns array of probabilities evaluated on a grid of N_mag x N_z x N_mh"""
    if do_abs:
        probs_fn = path.join(data_dir, output_tag + '_muvz_probs_abs.npy')
    else:
        probs_fn = path.join(data_dir, output_tag + '_muvz_probs.npy')
    if path.isfile(probs_fn) and not recompute:
        probs = np.load(probs_fn)
    else:
        mag_grid = mag_grid.reshape(-1,1,1)
        mean = mean.reshape(1, nz, nm)
        std = std.reshape(1, nz, nm)
        mag_dim = len(mag_grid)
        probs = np.exp(-(mag_grid-mean)**2/2.0/std**2)/np.sqrt(2*np.pi)/std
        
        for i, mag in enumerate(mag_grid):
            if do_abs:
                idx = mag > (maxs * 0.9)
            else:
                idx = mag > (maxs * 1.05)
            probs[i][idx] = 0
            if do_abs:
                idx = mag < (mins * 1.1)
            else:
                idx = mag < (mins * 0.95)
            probs[i][idx] = 0

        # norm = np.sum(probs, axis=(0,1)) 
        # norm = norm.reshape(1, 1, nm)
        # np.isclose(norm, )
        # print(norm <= 0)
        
        # probs = (probs / norm)
        P_mh = bweights / np.sum(bweights, axis=1).reshape(1,-1,1)
        # print(P_mh.shape)
        probs *= P_mh
        # ngdeep_eff_volume, ceers_eff_volume
        probs = np.sum(probs, axis=2)
        ngdeep_probs = probs * ngdeep_eff_volume
        ngdeep_probs /= np.sum(ngdeep_probs)
        ceers_probs = probs * ceers_eff_volume
        ceers_probs /= np.sum(ceers_probs)
        # phi = np.sum(bweights * probs, axis=2)
        if save:
            ngdeep_fn = path.join(data_dir, output_tag + '_ngdeep_muvz_probs.npy')
            ceers_fn = path.join(data_dir, output_tag + '_ceers_muvz_probs.npy')
            np.save(ngdeep_fn, ngdeep_probs)
            np.save(ceers_fn, ceers_probs)
    return probs


def plot_data(logmhs, mags, zs, data_dir, do_abs):
    plot_zs = [8.0, 12.0, 16.0]
    f, axs = plt.subplots(len(plot_zs), 1, figsize=(6,15),constrained_layout=True, sharex=True)
    for i,z in enumerate(plot_zs):
        ax = axs[i]
        idx = zs==z
        mhs = logmhs[idx]
        muvs = mags[idx]
        ax.scatter(mhs, muvs, marker='x',c='k', label="Sim. data")
        if do_abs:
            ax.set_ylim(0,-25)
            ax.set_ylabel(r'$M_{\mathrm{UV}}$')
        else:
            ax.set_ylim(35,22)
            ax.set_ylabel(r'$m_{\mathrm{UV}}$')
    ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
    if do_abs:
        plt.savefig(path.join(data_dir,'abs_data.pdf'))
    else:
        plt.savefig(path.join(data_dir,'app_data.pdf'))
    plt.close('all')


def plot_probs(logmhs, mags, zs, means, sigmas, mins, maxs, data_dir, output_tag, do_abs):
    print('plot probs')
    plot_zs = [12.0]
    # f, axs = plt.subplots(1, len(plot_zs), figsize=(10,5),constrained_layout=True, sharey=True)
    f, ax = plt.subplots(1, len(plot_zs), figsize=(5,5),constrained_layout=True, sharey=True)
    colors = sns.color_palette("Blues",n_colors=2)
    if do_abs:
        mag_grid = np.linspace(-25.0, 0.0, 1000)
    else:
        mag_grid = np.linspace(22.0, 45.0, 1000)
    probs = get_probs(mag_grid, means, sigmas, mins, maxs, data_dir, output_tag, do_abs, True, False)
    for i,z in enumerate(plot_zs):
        # ax = axs[i]
        idx = zs==z
        mhs = logmhs[idx]
        muvs = mags[idx]
        # means = means[idx,:]
        # sigmas = sigmas[idx,:]
        
        ax.scatter(mhs, muvs, marker='x',c='k', label="Sim. data")

        pidx = z_grid == z
        probs_z = probs[:,pidx,:].reshape(len(mag_grid),nm)
        peak = []
        one_sigma_upper = []
        two_sigma_upper = []
        one_sigma_lower = []
        two_sigma_lower = []
        for j in range(nm):
            p = probs_z[:,j]
            pj = np.argmax(p)
            pmax = np.amax(p)
            p/=pmax
            peak.append(mag_grid[pj])

            upper = p[pj+1:]
            upper = upper[np.nonzero(upper)]
            if len(upper)==0: 
                one_sigma_upper.append(mag_grid[pj])
                two_sigma_upper.append(mag_grid[pj])
            else:
                osui = np.argmin(np.abs(upper-np.exp(-0.5)))
                one_sigma_upper.append(mag_grid[osui+pj])
                tsui = np.argmin(np.abs(upper-np.exp(-2.0)))
                two_sigma_upper.append(mag_grid[tsui+pj])

            lower = p[:pj] 
            iszero = np.isclose(lower, 0)
            nz = np.sum(iszero)
            lower = lower[np.logical_not(iszero)]
            if len(lower)==0:
                one_sigma_lower.append(mag_grid[pj])
                two_sigma_lower.append(mag_grid[pj])
            else:
                osli = np.argmin(np.abs(lower-np.exp(-0.5)))
                one_sigma_lower.append(mag_grid[nz+osli])
                tsli = np.argmin(np.abs(lower-np.exp(-2.0)))
                two_sigma_lower.append(mag_grid[nz+tsli])
            

        # one_sigma_upper = np.argmin(np.abs(probs_z+np.exp(-0.5)),axis=0)
        # one_sigma_lower = np.argmin(np.abs(probs_z-np.exp(-0.5)),axis=0)
        # two_sigma_lower = np.argmin(np.abs(probs_z-np.exp(-2.0)),axis=0)
        # two_sigma_upper = np.argmin(np.abs(probs_z+np.exp(-2.0)),axis=0)
        peak = np.array(peak)
        ngdp_idx = np.argmin(np.abs(peak-30.4))
        mh_ngdp = bin_centers[ngdp_idx]
        ceers_idx = np.argmin(np.abs(peak-29.15))
        mh_ceers = bin_centers[ceers_idx]

        reds = sns.color_palette("Reds",n_colors=2)
        ax.hlines(30.4, 8.0,  11.5, linestyle='--', color=reds[1], label='NGDEEP $5\sigma$ depth', zorder=8)
        ax.arrow(11.1, 30.4, 0, -0.2, color=reds[1], linewidth=1.5, head_width=0.03, head_length=0.05)
        # ax.vlines(mh_ngdp, 35, 30.4, color=reds[1], zorder=8)
        ax.hlines(29.15, 8.0, 11.5, linestyle='--', color=reds[0], label='CEERS $5\sigma$ depth', zorder=7)
        ax.arrow(11.1, 29.15, 0, -0.2, color=reds[0], linewidth=1.5, head_width=0.03, head_length=0.05)
        # ax.vlines(mh_ceers, 35, 29.15, color=reds[0], zorder=7)

        ax.fill_between(bin_centers, two_sigma_lower, two_sigma_upper, alpha=0.65, color=colors[0], zorder=5)
        ax.fill_between(bin_centers, one_sigma_lower, one_sigma_upper, alpha=0.85, color=colors[1], zorder=6)
        # ax.plot(bin_centers, peak, 'k')
        ax.set_title(f'$z={z}$')
        if do_abs:
            ax.set_ylim(-16.5,-23.5)
            if i==0:
                ax.set_ylabel(r'$M_{\mathrm{UV}}$')
        else:
            ax.set_ylim(31.0,24.5)
            if i==0:
                ax.set_ylabel(r'$m_{\mathrm{UV}}$')
        ax.set_xlim(8.9, 11.5)
        ax.legend(frameon=False, fontsize=12)
    ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\,/\,M_{\odot}\right)$')
    if do_abs:
        plt.savefig(path.join(data_dir, output_tag+'_abs_probs.pdf'))
    else:
        plt.savefig(path.join(data_dir, output_tag+'_app_probs.pdf'))
        print('saving '+output_tag+'_app_probs.pdf')
    plt.close('all')


def get_uvlf(probs, bweights, data_dir, output_tag, do_abs, recompute):
    if do_abs:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf_abs.npy')
    else:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf.npy')
    if path.isfile(uvlf_fn) and not recompute:
        phi = np.load(uvlf_fn)
    else:
        phi = np.sum(bweights * probs, axis=2)
        np.save(uvlf_fn, phi)
    return phi


def plot_uvlf(muv_grid, z_grid, uvlf, volumes, data_dir):
    c = plt.cm.Dark2(3)
    # zs = [8.0, 12.0, 16.0]
    app_cutoff = 30.4
    f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)

    dz = z_grid[1]-z_grid[0]
    z_left = z_grid-dz/2.0
    z_right = z_grid+dz/2.0

    ax = axs[0]
    zidx = (z_left > 8.5) & (z_right < 9.5)
    zs = z_grid[zidx]
    z_left_slice = z_left[zidx]
    z_right_slice = z_right[zidx]
    zuvlf = uvlf[:,zidx]
    zvolume = volumes[zidx]
    totalV = np.zeros_like(muv_grid)
    y = np.zeros_like(muv_grid)
    for i,muv in enumerate(muv_grid):
        for j in range(len(zs)):
            # left > right
            # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
            right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[j]).value + 2.5*np.log10(1+z_right_slice[j])
            if muv < right_cutoff:
                y[i] += zuvlf[i,j]*zvolume[j]
                totalV[i] += zvolume[j]
            # elif muv < left_cutoff:
            #     volume = 
    y /= totalV

    # trying to make things apples to apples !!!
    # plot_muvs = np.arange(-23.,-16.05,0.5)
    # left = plot_muvs[:-1]
    # right = plot_muvs[1:]
    # center = (left+right)/2.0
    # new_dmuv = right[0]-left[0] 
    # old_dmuv = muv_grid[1]-muv_grid[0]
    # new_y = np.zeros_like(center)
    # for i,muv in enumerate(center):
    #     bidx = (muv_grid >= left) & (muv_grid < right)
    #     new_y = np.sum(y[bidx])*old_dmuv/new_dmuv

    midx = y>0
    y = np.log10(y[midx])
    ax.plot(muv_grid[midx], y, label='Sim.', lw=2.5, c=c)

    # for testing purposes only
    # zidx = z_grid == 8.0
    # y = uvlf[:,zidx].flatten()
    # midx = y>0
    # y = np.log10(y[midx])
    # ax.plot(muv_grid[midx],y, 'k--', label='z=8.0')

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
    # print(ceers_muv_z9[1:].shape, ceers_log_phi[not_nan_idx].shape, ceers_log_err2.shape)
    

    ax.scatter(ngdeep_muv, ngdeep_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',  label='NGDEEP (Leung et. al. 2023)')
    # ax.errorbar(ngdeep_muv, obs_log_phi, elinewidth=1.8, yerr=ngdeep_log_err, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar(ngdeep_muv, ngdeep_log_phi, yerr=ngdeep_log_err2, 
                elinewidth=1.8,  marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar([-21.1],[np.log10(8.9e-5)], yerr=ngdeep_log_err[0][0], uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z9, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    # ax.errorbar(ceers_muv_z9, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.errorbar(ceers_muv_z9[1:], ceers_log_phi[1:], yerr=ceers_log_err2,
                marker='o', color="none", ecolor='gray', capsize=3.5, ls='none',elinewidth=1.8) #  
    ax.errorbar(ceers_muv_z9[0], ceers_log_phi[0], yerr=[[ceers_log_err[0][0]],[ceers_log_upper_err2[0]]], #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8) # ls='none'
    ax.errorbar([-22.5, ceers_muv_z9[0],-21.5], [np.log10(0.9e-5),ceers_log_phi[0],np.log10(0.9e-5)], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 

    ax.set_title(r'$8.5<z<9.5$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_xlim(-23.0, -16.5)
    ax.set_ylim(-7, -1.5)
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    
    ax = axs[1]
    zidx = (z_left > 9.5) & (z_right < 12.0)
    zs = z_grid[zidx]
    z_left_slice = z_left[zidx]
    z_right_slice = z_right[zidx]
    zuvlf = uvlf[:,zidx]
    zvolume = volumes[zidx]
    totalV = np.zeros_like(muv_grid)
    y = np.zeros_like(muv_grid)
    for i,muv in enumerate(muv_grid):
        for j in range(len(zs)):
            # left > right
            # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
            right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[j]).value + 2.5*np.log10(1+z_right_slice[j])
            if muv < right_cutoff:
                y[i] += zuvlf[i,j]*zvolume[j]
                totalV[i] += zvolume[j]
            # elif muv < left_cutoff:
            #     volume = 
    y /= totalV
    # y = np.sum(uvlf[:,zidx]*volumes[zidx],axis=1)
    # y /= np.sum(volumes[zidx])
    midx = y>0
    y = np.log10(y[midx])
    ax.plot(muv_grid[midx], y, label='Sim.', lw=2.5, c=c)

    # for testing purposes only
    # zidx = z_grid == 12.0
    # y = uvlf[:,zidx].flatten()
    # midx = y>0
    # y = np.log10(y[midx])
    # ax.plot(muv_grid[midx],y, 'k--', label='z=12.0')

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
    # ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err2, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.errorbar([-20.05],[np.log10(9.7e-5)], yerr=log_err[0][0], uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z11, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    # ax.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.errorbar(ceers_muv_z11[1:][:-1], ceers_log_phi[1:][:-1], elinewidth=1.8, yerr=ceers_log_err2, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.errorbar([ceers_muv_z11[0],ceers_muv_z11[-1]], [ceers_log_phi[0],ceers_log_phi[-1]], yerr=[[ceers_log_err[0][0],ceers_log_err[0][-1]],[ceers_log_err2[1][0],ceers_log_err2[1][-1]]], #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') # ls='none'
    ax.errorbar([-21.0, -20.5,ceers_muv_z11[-1]], [np.log10(0.5e-5),ceers_log_phi[0],ceers_log_phi[-1]], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 

    ax.set_title(r'$9.5<z<12.0$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(-23.0, -16.5)
    ax.set_ylim(-7, -1.5)
    plt.savefig(path.join(data_dir, 'model_uvlf.pdf'))
    plt.close('all')


def plot_uvlf_z(muv_grid, z_grid, uvlf, volumes, data_dir):
    c = plt.cm.Dark2(3)
    # zs = [8.0, 12.0, 16.0]
    f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)

    ax = axs[0]
    # zidx = (z_grid > 8.5) & (z_grid < 9.5)
    # y = np.sum(uvlf[:,zidx]*volumes[zidx],axis=1)
    # y /= np.sum(volumes[zidx]) 
    colors = sns.color_palette("flare",n_colors=6)
    
    for i,z in enumerate([8.5,9.0,9.5]):
        zidx = z_grid == z
        y = uvlf[:,zidx].flatten()
        midx = y>0
        y = np.log10(y[midx])
        ax.plot(muv_grid[midx],y, c=colors[i], label=f'z={z}', lw=2.0)
        

    ngdeep_muv = [-20.1, -19.1, -18.35, -17.85, -17.35]
    ngdeep_phi = np.array([14.7e-5, 18.9e-5, 74.0e-5, 170.0e-5, 519.0e-5])
    ngdeep_phi_err = np.array([[7.2e-5, 8.9e-5, 29.0e-5, 65.e-5, 198.e-5],
                      [11.1e-5, 13.8e-5, 41.4e-5, 85.e-5, 248.e-5]])
    obs_log_phi = np.log10(ngdeep_phi)
    log_err = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-obs_log_phi]
    
    ceers_muv_z9 = [-22.0, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5]
    ceers_phi_z9 = np.array([1.1e-5, 2.2e-5, 8.2e-5, 9.6e-5, 28.6e-5, 26.8e-5, 136.0e-5])
    ceers_log_phi = np.log10(ceers_phi_z9)
    ceers_upper_z9 = np.array([0.7e-5, 1.3e-5, 4.0e-5, 4.6e-5, 11.5e-5, 12.4e-5, 61.0e-5])
    ceers_lower_z9 = np.array([0.6e-5, 1.0e-5, 3.2e-5, 3.6e-5, 9.1e-5, 10.0e-5,49.9e-5])
    ceers_log_err = [ceers_log_phi-np.log10(ceers_phi_z9-ceers_lower_z9),np.log10(ceers_phi_z9+ceers_upper_z9)-ceers_log_phi]
    
    ax.scatter(ngdeep_muv, obs_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',  label='NGDEEP (Leung et. al. 2023)')
    ax.errorbar(ngdeep_muv, obs_log_phi, elinewidth=1.8, yerr=log_err, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z9, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    ax.errorbar(ceers_muv_z9, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.set_title(r'$8.5<z<9.5$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_xlim(-22.5, -16.5)
    ax.set_ylim(-7, -1.5)
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    
    ax = axs[1]
    for i,z in enumerate([9.5,10.75,12.0]):
        zidx = np.argmin(np.abs(z_grid-z)) # z_grid == z
        # print(z_g/rid[zidx])
        z = z_grid[zidx]
        y = uvlf[:,zidx].flatten()
        midx = y>0
        y = np.log10(y[midx])
        ax.plot(muv_grid[midx],y, c=colors[i+3], label=f'z={z}', lw=2.0)

    # for testing purposes only
    # zidx = z_grid == 12.0
    # y = uvlf[:,zidx].flatten()
    # midx = y>0
    # y = np.log10(y[midx])
    # ax.plot(muv_grid[midx],y, 'k--', label='z=12.0')

    ngdeep_muv = [-19.35, -18.65, -17.95, -17.25]
    ngdeep_phi = np.array([18.5e-5, 27.7e-5, 59.1e-5, 269.0e-5])
    ngdeep_phi_err = np.array([[8.3e-5, 13.0e-5, 29.3e-5, 124.e-5],
                    [11.9e-5, 18.3e-5, 41.9e-5, 166.e-5]])
    obs_log_phi = np.log10(ngdeep_phi)
    log_err = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-obs_log_phi]

    ceers_muv_z11 = [-20.5, -20.0, -19.5, -19.0, -18.5]
    ceers_phi_z11 = np.array([1.8e-5, 5.4e-5, 7.6e-5, 17.6e-5, 26.3e-5])
    ceers_log_phi = np.log10(ceers_phi_z11)
    ceers_upper_z11 = np.array([1.2e-5, 2.7e-5, 3.9e-5, 10.3e-5, 18.2e-5])
    ceers_lower_z11 = np.array([0.9e-5, 2.1e-5, 3.0e-5, 7.9e-5, 13.3e-5])
    ceers_log_err = [ceers_log_phi-np.log10(ceers_phi_z11-ceers_lower_z11),np.log10(ceers_phi_z11+ceers_upper_z11)-ceers_log_phi]
    

    ax.scatter(ngdeep_muv, obs_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k')
    ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, label='NGDEEP data (Leung et. al. 2023)', marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z11, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    ax.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.set_title(r'$9.5<z<12.0$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlim(-22.5, -16.5)
    ax.set_ylim(-7, -1.5)
    plt.savefig(path.join(data_dir, 'model_uvlf_z.pdf'))
    plt.close('all')


def plot_like_bins(data_dir, app_uvlf):
    z_cutoff = 8.5
    zidx = z_grid >= z_cutoff
    midx = (app_grid < 31) & (app_grid >= 25)
    app_uvlf = app_uvlf[midx,:][:,zidx]
    # ngdeep_eff_volume = ngdeep_eff_volume[midx,:][:,zidx]
    # ceers_eff_volume = ceers_eff_volume[midx,:][:,zidx]
    ngdeep_sim_n = app_uvlf*ngdeep_eff_volume[midx,:][:,zidx]
    ceers_sim_n = app_uvlf*ceers_eff_volume[midx,:][:,zidx]
    sim_n = ngdeep_sim_n+ceers_sim_n
    print(f'ngdeep total = {np.sum(ngdeep_sim_n)}, ceers total = {np.sum(ceers_sim_n)}')
    ceers_data_n = np.sum(ceers_pdf[:,midx,:][:,:,zidx], axis=0)
    ngdeep_data_n = np.sum(ngdeep_pdf[:,midx,:][:,:,zidx], axis=0)
    data_n = ceers_data_n+ngdeep_data_n
    # fig, axs = plt.subplots(1, 3, figsize=(15,5),layout='constrained')
    fig = plt.figure(figsize=(12,12),layout='constrained')
    gs = gridspec.GridSpec(2, 4, figure=fig,left=0.05,right =.95 ,wspace=0.05, hspace=0.05) # left=0, right=1, wspace=0.1, hspace=0.1
    norm = mpl.colors.Normalize(vmin=0, vmax=np.amax([data_n,sim_n]))

    ax = fig.add_subplot(gs[0, :2])
    # ax = axs[0]
    ax.imshow(data_n, cmap='Greys', norm=norm, aspect='auto')
    ax.set_ylabel(r'$m_{\mathrm{uv}}$')
    yticks = range(len(app_grid[midx]))[::4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(app_grid[midx][::4])
    xticks = range(len(z_grid))[::4]
    ax.set_xticks(xticks)
    ax.set_xticklabels(z_grid[::4])
    ax.set_xlabel('$z$')
    ax.set_title('Data counts')

    ax = fig.add_subplot(gs[0, 2:])
    # ax = axs[1]
    ax.imshow(sim_n, cmap='Greys', norm=norm, aspect='auto')
    ax.set_ylabel(r'$m_{\mathrm{uv}}$')
    yticks = range(len(app_grid[midx]))[::4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(app_grid[midx][::4])
    xticks = range(len(z_grid))[::4]
    ax.set_xticks(xticks)
    ax.set_xticklabels(z_grid[::4])
    ax.set_xlabel('$z$')
    ax.set_title('Sim. counts')

    ax = fig.add_subplot(gs[1, 1:3])
    # ax = axs[2]
    ax.imshow(np.abs(sim_n-data_n), cmap='Greys', norm=norm, aspect='auto')
    ax.set_ylabel(r'$m_{\mathrm{uv}}$')
    yticks = range(len(app_grid[midx]))[::4]
    ax.set_yticks(yticks)
    ax.set_yticklabels(app_grid[midx][::4])
    xticks = range(len(z_grid))[::4]
    ax.set_xticks(xticks)
    ax.set_xticklabels(z_grid[::4])
    ax.set_xlabel('$z$')
    ax.set_title('|Sim. counts - data counts|')

    plt.savefig(path.join(data_dir, 'binned_counts.pdf'))
    plt.close('all')


def plot_Mh_given_fixed_Muv(data_dir, abs_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""
    f, axs = plt.subplots(3, 1, figsize=(6,15), constrained_layout=True)
    P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
    prob_mh_given_Muv = abs_probs *  P_mh
    prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
    # print(prob_mh_given_Muv.shape)
    for i,z in enumerate([8.0,12.0,16.0]):
        zidx = z_grid==z
        for j,M_uv in enumerate([-17, -19, -21]):
            c = plt.cm.Dark2(j)
            midx = abs_grid==M_uv
            prob = prob_mh_given_Muv[midx, zidx]
            axs[i].plot(bin_centers, prob.flatten(), c=c, label=r'$M_{\mathrm{uv}}='+str(M_uv)+'$')
        axs[i].set_ylabel(r'$P(M_{h}|M_{\mathrm{uv}})$')
        axs[i].set_ylim(0,0.25)
        axs[i].set_xlim(8.5, 11.25)
        axs[i].set_title(f'$z={z}$')
        axs[i].legend(frameon=False, fontsize=13)
    axs[-1].set_xlabel(r'$M_{h}$')

    plt.savefig(path.join(data_dir, 'prob_mh_given_muv.pdf'))
    plt.close('all')
    

def plot_Mh_from_data(data_dir, app_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""
    # f, axs = plt.subplots(3, 1, figsize=(6,15), constrained_layout=True)
    zidx = z_grid >= 8.5
    P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
    prob_mh_given_Muv = app_probs *  P_mh
    prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
    ng, nmuv, nz = ngdeep_pdf.shape
    # print(ng, nmuv, nz)
    # print(prob_mh_given_Muv.shape)
    ngd_pdf = ngdeep_pdf.reshape(ng,nmuv,nz,1)
    c_pdf = ceers_pdf.reshape(-1,nmuv,nz,1)
    prob_mh_given_Muv = prob_mh_given_Muv.reshape(1, nmuv, nz, -1)
    
    prob_mh = np.sum(ngd_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
    prob_mh += np.sum(c_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
    prob_mh /= np.sum(prob_mh)
    plt.plot(bin_centers, prob_mh)
    plt.ylabel(r'$P(M_{h}|\mathcal{D})$')
    # plt.ylim(0,0.25)
    plt.xlim(8.5, 11.25)
    # axs[i].set_title(f'$z={z}$')
    plt.xlabel(r'$M_{h}$')
    plt.tight_layout()
    plt.savefig(path.join(data_dir, 'prob_mh_given_data.pdf'))
    plt.close('all')


def plot_Mh_from_data_z(data_dir, app_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""
    # f, axs = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True, sharey=True)
    interp_mh = np.linspace(8.0, 11.5, 1000)

    for i in range(3):
        i = 2-i
        # ax = axs[i]
        if i==0:
            zidx = (z_grid >= 8.5) & (z_grid<9.5)
        elif i==1:
            zidx = (z_grid >= 9.5) & (z_grid<12.0)
        else:
            zidx = (z_grid >= 8.5)

        P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
        prob_mh_given_Muv = app_probs *  P_mh
        prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
        ng, nmuv, nz = ngdeep_pdf.shape
        # print(ng, nmuv, nz)
        # print(prob_mh_given_Muv.shape)
        ngd_pdf = ngdeep_pdf.reshape(ng,nmuv,nz,1)
        c_pdf = ceers_pdf.reshape(-1,nmuv,nz,1)
        prob_mh_given_Muv = prob_mh_given_Muv.reshape(1, nmuv, nz, -1)
        
        prob_mh = np.sum(ngd_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
        prob_mh += np.sum(c_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
        prob_mh /= np.max(prob_mh) # np.sum(prob_mh)

        prob_mh = np.interp(interp_mh, bin_centers, prob_mh)

        if i==0:
            label = r'$8.5 \leq z < 9.5$'
        elif i==1:
            label = r'$9.5 \leq z < 12.0$'
        else:
            label = 'All $z$'
        if i==2:
            c='k'
            p1 = np.e**(-0.5)
            p2 = np.e**(-2.0)
            idx1 = prob_mh >= p1
            idx2 = (prob_mh >= p2)# & (prob_mh <= p1)
            plt.fill_between(interp_mh[idx1],0, prob_mh[idx1],color=c, alpha=0.7)
            plt.fill_between(interp_mh[idx2],0, prob_mh[idx2], color=c, alpha=0.5)
        else:
            c=plt.cm.Dark2((i+1)*3)
        plt.plot(interp_mh, prob_mh, c=c, label=label)
        
    plt.xlim(8.5, 11.25)
    plt.xlabel(r'$\mathrm{Log}(M_{h})$')
    plt.ylabel(r'$P(M_{h}|\mathcal{D})$')
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=12)
    plt.savefig(path.join(data_dir, 'prob_mh_given_data_z.pdf'))
    plt.close('all')


def linear_interpolate(x1,y1,x2,y2):
    slope = (y2-y1)/(x2-x1)
    intercept = y2-(y2-y1)/(x2-x1) * x2
    return slope, intercept


def like_function(app_uvlf):
    z_cutoff = 8.5
    zidx = z_grid >= z_cutoff
    ngdeep_n = np.sum(app_uvlf[:,zidx]*ngdeep_eff_volume[:,zidx], dtype=np.longdouble) 
    ceers_n = np.sum(app_uvlf[:,zidx]*ceers_eff_volume[:,zidx], dtype=np.longdouble) 
    print(f'ngdeep total = {ngdeep_n}, ceers total = {ceers_n}')
    loglike = -ngdeep_n-ceers_n
    ngdeep_obs_n = np.sum(ngdeep_pdf[:,:,zidx]*app_uvlf[:,zidx]*ngdeep_eff_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ngdeep_obs_n = np.log(ngdeep_obs_n[ngdeep_obs_n>0])
    ngdeep_obs_n = np.sum(ngdeep_obs_n)
    loglike += ngdeep_obs_n
    ceers_obs_n = np.sum(ceers_pdf[:,:,zidx]*app_uvlf[:,zidx]*ceers_eff_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ceers_obs_n = np.log(ceers_obs_n[ceers_obs_n>0])
    ceers_obs_n = np.sum(ceers_obs_n)
    loglike += ceers_obs_n
    print(f'loglike={loglike}')
    return loglike


def cv_like_function(app_uvlf):
    z_cutoff = 8.5
    zidx = z_grid >= z_cutoff
    ngd_pdf = ngdeep_pdf[:,:,zidx]
    n_ngd = ngd_pdf.shape[0]
    c_pdf = ceers_pdf[:,:,zidx]
    n_ceers = c_pdf.shape[0]
    # ngdeep_n = np.sum(app_uvlf[:,zidx]*ngdeep_eff_volume[:,zidx], dtype=np.longdouble) 
    # ceers_n = np.sum(app_uvlf[:,zidx]*ceers_eff_volume[:,zidx], dtype=np.longdouble) 
    # print(f'ngdeep total = {ngdeep_n}, ceers total = {ceers_n}')
    # loglike = -ngdeep_n-ceers_n
    mu = app_uvlf[:,zidx]*ngdeep_eff_volume[:,zidx]
    sigma = mu + mu**2 * cv[zidx].reshape(1,-1)
    # print(mu)
    k = (mu/sigma)**2
    # print(k)
    theta = sigma**2/mu
    # print(theta)
    # non_nan_idx = np.logical_not(np.logical_or(np.isnan(theta),np.isnan(k)))
    # print(non_nan_idx.shape, k.shape)
    # k = k[non_nan_idx]
    # theta = theta[non_nan_idx]
    like_integrand = np.exp(-1./theta)/theta**k/gamma(k)
    idx = np.isnan(like_integrand)
    like_integrand[idx] = 0
    # ngd_bool = np.tile(non_nan_idx,(n_ngd, 1, 1))
    # ceers_bool = np.tile(non_nan_idx,(n_ceers, 1, 1))
    # print(ngd_pdf.shape)
    # print(ngd_bool.shape)

    ngdeep_obs_n = np.sum(ngd_pdf*like_integrand, axis=(1,2), dtype=np.longdouble) 
    ngdeep_obs_n = np.log(ngdeep_obs_n[ngdeep_obs_n>0])
    ngdeep_obs_n = np.sum(ngdeep_obs_n)
    loglike = ngdeep_obs_n
    ceers_obs_n = np.sum(c_pdf*like_integrand, axis=(1,2), dtype=np.longdouble) 
    ceers_obs_n = np.log(ceers_obs_n[ceers_obs_n>0])
    ceers_obs_n = np.sum(ceers_obs_n)
    loglike += ceers_obs_n
    print(f'loglike={loglike}')
    return loglike


def binned_like_function(app_uvlf):
    z_cutoff = 8.5
    zidx = z_grid >= z_cutoff
    ngdeep_sim_n = app_uvlf[:,zidx]*ngdeep_eff_volume[:,zidx]
    ceers_sim_n = app_uvlf[:,zidx]*ceers_eff_volume[:,zidx]
    print(f'ngdeep total = {np.sum(ngdeep_sim_n)}, ceers total = {np.sum(ceers_sim_n)}')
    ceers_data_n = np.sum(ceers_pdf[:,:,zidx], axis=0)
    ngdeep_data_n = np.sum(ngdeep_pdf[:,:,zidx], axis=0)
    # print(ceers_data_n.shape, ceers_sim_n.shape)
    
    # print(scipy.special.gamma(ceers_data_n+1.0))
    # ngdeep_sim_n = np.where(ngdeep_sim_n<=0., 1e-50, ngdeep_sim_n) # fixes issues where sim_n = 0 -> like=0 -> loglike=inf
    # ceers_sim_n = np.where(ceers_sim_n<=0., 1e-50, ceers_sim_n)

    ngdeep_like = np.power(ngdeep_sim_n,ngdeep_data_n) * np.exp(-ngdeep_sim_n) / scipy.special.gamma(ngdeep_data_n+1.0)
    # print(np.sum(ngdeep_like<=0.))
    ngdeep_loglike = np.log(np.prod(ngdeep_like,dtype=np.longdouble))
    # print(np.sum(ngdeep_like<=0))
    ceers_like = np.power(ceers_sim_n,ceers_data_n) * np.exp(-ceers_sim_n) / scipy.special.gamma(ceers_data_n+1.0)
    ceers_loglike = np.log(np.prod(ceers_like,dtype=np.longdouble))
    # print(ceers_loglike)
    loglike = ngdeep_loglike+ceers_loglike
    print(f'loglike={loglike}')
    return loglike

def get_halo_bias(hmf_fn):
    weights_fn = 'data/gpr_zgrid_bias.npy'
    if path.isfile(weights_fn):
        biases = np.load(weights_fn) 
    else:
        biases = []
        f = h5py.File(hmf_fn,"r")
        biases = []
        for i in range(17):
            j = 17-i
            output = f[f'Outputs/Output{j}']
            haloBias = output['haloBias'][:]
            biases.append(haloBias)
        biases = np.array(biases)
        f.close()
        np.save(weights_fn,biases)
    return biases

# def get_bias(uvlf, logmhs):
#     galaxy_bias = np.trapz(hmf*halo_bias*Prob_Muv_given_Mh, logmhs)
#     return galaxy_bias


ngdeep_data = pd.read_csv('/home/gdriskell/galacticus/data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('/home/gdriskell/galacticus/data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
# ceers_data = ceers_data.sort_values('mf277w', ascending=True)
zgrid_weights = get_weights_from_hmf('/home/gdriskell/galacticus/data/gpr_zgrid_hmfs:MPI0000.hdf5')


dMh = 0.15
Mh_bins = np.arange(8.0, 11.76, dMh)
global nm
nm = len(Mh_bins)-1
bin_centers = Mh_bins[:-1] + dMh/2.0 

global nz
nz = 17
global z_grid
z_grid = np.linspace(8.0, 16.0, nz)
dz = z_grid[1]-z_grid[0]

cv_z = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]
cv = [0.1522, 0.17411, 0.18843, 0.22613, 0.2549, 0.26993, 0.292834, 0.3769, 0.426021, 0.43414, 0.49211, 0.59527, 0.664436, 0.62814, 0.602, 0.602]
cv = np.interp(z_grid, cv_z, cv)

# print(cv)

logmhs = np.log10(np.geomspace(1.0e8,5.0e11,3699))
global binned_weights
binned_weights = get_binned_weights(Mh_bins,logmhs,zgrid_weights, True)
# print(binned_weights)

abs_min = -25.0
abs_max = 0.
dabs = 0.2
nabs = int(round((abs_max-abs_min)/dabs))+1
abs_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = abs_grid[1]-abs_grid[0]

app_min = 22.0
app_max = 45.0
dapp = 0.25 #0.2
napp = int(round((app_max-app_min)/dapp))+1
app_grid = np.linspace(app_min, app_max, napp) 
dapp = app_grid[1]-app_grid[0]

# global t
t_grid = cosmo.age(z_grid)
t8 = cosmo.age(8.0)
t12 = cosmo.age(12.0)
t16 = cosmo.age(16.0)

z_volumes = (cosmo.comoving_volume(z_grid+dz/2.0)-cosmo.comoving_volume(z_grid-dz/2.0)).value 

ngdeep_cf = get_ngdeep_cf(app_grid)
ngdeep_area = 3.3667617763401435e-08 # 5 arcmin^2 as a fraction of the sky
# print(ngdeep_area)
ngdeep_volumes = ngdeep_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
ngdeep_eff_volume = ngdeep_cf.reshape(-1,1)*ngdeep_volumes.reshape(1,-1)
ngdeep_eff_volume = np.abs(ngdeep_eff_volume)

ceers_cf = get_ceers_cf(app_grid)
ceers_area = 5.932234249911333e-07 # 88.1 arcmin^2 as a fraction of the sky
# print(ceers_area)
ceers_volumes = ceers_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
ceers_eff_volume = ceers_cf.reshape(-1,1)*ceers_volumes.reshape(1,-1)
ceers_eff_volume = np.abs(ceers_eff_volume)

# print(effective_volume)
ngdeep_pdf_fn =  '/home/gdriskell/galacticus/data/ngdeep_pdf.npy'
ceers_pdf_fn = '/home/gdriskell/galacticus/data/ceers_pdf.npy'
ngdeep_pdf = np.abs(get_data_pdf(ngdeep_data, app_grid, z_grid, ngdeep_pdf_fn, True))
ceers_pdf = np.abs(get_data_pdf(ceers_data, app_grid, z_grid, ceers_pdf_fn, True))


def evaluate_likelihood(i, data_dir, reload, recompute, plot, do_like):
    output_tag = 'interp'
    # data_dir = f'/data001/gdriskell/jwst_blitz_astro_samples/updated_params_p{i}/'
    weights, logmhs, abs_mags, app_mags, zs = load_data(data_dir, reload)
    #print(len(logmhs), len(abs_mags), len(app_mags), len(zs))
    data_zs = np.unique(zs)
    n_dz = len(data_zs)
    abs_maxs = np.zeros((n_dz,nm))
    abs_mins = np.zeros((n_dz,nm))
    abs_means = np.zeros((n_dz,nm))
    abs_stds = np.zeros((n_dz,nm))
    # app_means = np.zeros((n_dz,nm))
    # app_stds = np.zeros((n_dz,nm))
    app_maxs = np.zeros((n_dz,nm))
    app_mins = np.zeros((n_dz,nm))
    # plot_data(logmhs, abs_mags, zs, data_dir, True)
    if recompute:
        for i,z in enumerate(data_zs):
            # abs
            zidx = zs==z
            z_lmh = logmhs[zidx]
            z_abs = abs_mags[zidx]
            z_app = app_mags[zidx]

            for j in range(nm):
                # print(j)
                left = Mh_bins[j]
                right = Mh_bins[j+1]
                bidx = (z_lmh > left) & (z_lmh < right)
                if np.sum(bidx)==0:
                    abs_means[i,j] = -3
                    abs_stds[i,j] = 1
                    abs_maxs[i,j] = -1
                    abs_mins[i,j] = -5
                    app_maxs[i,j] = 45
                    app_mins[i,j] = 35
                else:
                    abs_means[i,j] = np.average(z_abs[bidx])
                    abs_stds[i,j] = np.std(z_abs[bidx])
                    abs_maxs[i,j] = np.amax(z_abs[bidx])
                    abs_mins[i,j] = np.amin(z_abs[bidx])

                    # app_means[i,j] = np.average(z_app[bidx])
                    # app_stds[i,j] = np.std(z_app[bidx])
                    app_maxs[i,j] = np.amax(z_app[bidx])
                    app_mins[i,j] = np.amin(z_app[bidx])

        sigma_abs_z = np.zeros((nz,nm))
        mean_abs_z = np.zeros((nz,nm))
        max_abs_z = np.zeros((nz,nm))
        min_abs_z = np.zeros((nz,nm))

        sigma_app_z = np.zeros((nz,nm))
        mean_app_z = np.zeros((nz,nm))
        max_app_z = np.zeros((nz,nm))
        min_app_z = np.zeros((nz,nm))

        mean_slope_abs_z8_z12, mean_intercept_abs_z8_z12 = linear_interpolate(t8, abs_means[0,:], t12, abs_means[1,:])
        std_slope_abs_z8_z12, std_intercept_abs_z8_z12 = linear_interpolate(t8, abs_stds[0,:], t12, abs_stds[1,:])
        max_slope_abs_z8_z12, max_intercept_abs_z8_z12 = linear_interpolate(t8, abs_maxs[0,:], t12, abs_maxs[1,:])
        min_slope_abs_z8_z12, min_intercept_abs_z8_z12 = linear_interpolate(t8, abs_mins[0,:], t12, abs_mins[1,:])

        max_slope_app_z8_z12, max_intercept_app_z8_z12 = linear_interpolate(t8, app_maxs[0,:], t12, app_maxs[1,:])
        min_slope_app_z8_z12, min_intercept_app_z8_z12 = linear_interpolate(t8, app_mins[0,:], t12, app_mins[1,:])

        mean_slope_abs_z12_z16, mean_intercept_abs_z12_z16 = linear_interpolate(t12, abs_means[1,:], t16, abs_means[2,:])
        std_slope_abs_z12_z16, std_intercept_abs_z12_z16 = linear_interpolate(t12, abs_stds[1,:], t16, abs_stds[2,:])
        max_slope_abs_z12_z16, max_intercept_abs_z12_z16 = linear_interpolate(t12, abs_maxs[1,:], t16, abs_maxs[2,:])
        min_slope_abs_z12_z16, min_intercept_abs_z12_z16 = linear_interpolate(t12, abs_mins[1,:], t16, abs_mins[2,:])

        max_slope_app_z12_z16, max_intercept_app_z12_z16 = linear_interpolate(t12, app_maxs[1,:], t16, app_maxs[2,:])
        min_slope_app_z12_z16, min_intercept_app_z12_z16 = linear_interpolate(t12, app_mins[1,:], t16, app_mins[2,:])

        for i,t in enumerate(t_grid):
            z = z_grid[i]
            if 8.0 <= z <= 12.0:
                mean_abs_z[i,:] = mean_slope_abs_z8_z12 * t + mean_intercept_abs_z8_z12
                sigma_abs_z[i,:] = std_slope_abs_z8_z12 * t + std_intercept_abs_z8_z12
                max_abs_z[i,:] = max_slope_abs_z8_z12 * t + max_intercept_abs_z8_z12
                min_abs_z[i,:] = min_slope_abs_z8_z12 * t + min_intercept_abs_z8_z12

                max_app_z[i,:] = max_slope_app_z8_z12 * t + max_intercept_app_z8_z12
                min_app_z[i,:] = min_slope_app_z8_z12 * t + min_intercept_app_z8_z12
            else:
                mean_abs_z[i,:] = mean_slope_abs_z12_z16 * t + mean_intercept_abs_z12_z16
                sigma_abs_z[i,:] = std_slope_abs_z12_z16 * t + std_intercept_abs_z12_z16
                max_abs_z[i,:] = max_slope_abs_z12_z16 * t + max_intercept_abs_z12_z16
                min_abs_z[i,:] = min_slope_abs_z12_z16 * t + min_intercept_abs_z12_z16

                max_app_z[i,:] = max_slope_app_z12_z16 * t + max_intercept_app_z12_z16
                min_app_z[i,:] = min_slope_app_z12_z16 * t + min_intercept_app_z12_z16
            mean_app_z[i,:] = mean_abs_z[i,:] + cosmo.distmod(z).value-2.5*np.log10(1.+z)
        sigma_app_z = sigma_abs_z.copy()
        abs_probs = get_probs(abs_grid, mean_abs_z, sigma_abs_z, min_abs_z, max_abs_z, data_dir, output_tag, True, recompute)
        app_probs = get_probs(app_grid, mean_app_z, sigma_app_z, min_app_z, max_app_z, data_dir, output_tag, False, recompute)
        full_app_probs = get_muvz_probs(app_grid, mean_app_z, sigma_app_z, min_app_z, max_app_z, binned_weights, data_dir, output_tag, False, recompute)
    else:
        abs_probs = get_probs(None, None, None, None, None, data_dir, output_tag, True, recompute)
        app_probs = get_probs(None, None, None, None, None, data_dir, output_tag, False, recompute)
    
    abs_uvlf = get_uvlf(abs_probs, binned_weights, data_dir, output_tag, True, recompute)/dabs
    app_uvlf = get_uvlf(app_probs, binned_weights, data_dir, output_tag, False, recompute)
    if plot:
        print('plotting')
        # plot_probs(logmhs, abs_mags, zs, mean_abs_z, sigma_abs_z, min_abs_z, max_abs_z, data_dir, output_tag, True)
        plot_probs(logmhs, app_mags, zs, mean_app_z, sigma_app_z, min_app_z, max_app_z, data_dir, output_tag, False)
        # plot_uvlf(abs_grid,z_grid,abs_uvlf,z_volumes,data_dir)
        
        # plot_Mh_from_data(data_dir, app_probs, binned_weights)
        plot_Mh_from_data_z(data_dir, app_probs, binned_weights)
        # plot_uvlf_z(abs_grid,z_grid,abs_uvlf,z_volumes,data_dir)
        # plot_probs(logmhs, app_mags, zs, app_grid, app_probs, data_dir, output_tag, False)
        # plot_like_bins(data_dir, app_uvlf)
        # plot_Mh_given_Muv(data_dir, abs_probs, binned_weights)
    if do_like:      
        # loglike = cv_like_function(app_uvlf)
        loglike = like_function(app_uvlf)
        # loglike = binned_like_function(app_uvlf)
    else: 
        loglike = None
    return loglike


def get_astro_params(data_dir):
    Vout_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
    alphaOut_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
    if 'final_params' in data_dir:
        tau0_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/timescale'
        alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/exponentVelocity'
    else:
        tau0_xml = 'starFormationRateDisks/starFormationTimescale/timescale'
        alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/exponentVelocity'
    xml_fn = path.join(data_dir,'z8.0.xml')
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    Vout = root.find(Vout_xml).get('value')
    alphaOut = root.find(alphaOut_xml).get('value')
    tau0 = root.find(tau0_xml).get('value')
    alphaStar = root.find(alphaStar_xml).get('value')
    return float(Vout), float(alphaOut), float(tau0), float(alphaStar)


def run(dirname, i, reload, recompute, plot, like):
    print(f'Starting p{i}')
    base = '/data001/gdriskell/jwst_blitz_astro_samples/'
    data_dir = path.join(base, dirname+f'_p{i}/')
    # logmhs, zs, app_mags, abs_mags = load_data(data_dir, reload).T
    # try:
    #     loglike = evaluate_likelihood(i, data_dir, reload, recompute, plot, like)
    # except:
    #     loglike = np.nan
    loglike = evaluate_likelihood(i, data_dir, reload, recompute, plot, like)
    Vout, alphaOut, tau0, alphaStar = get_astro_params(data_dir)
    return Vout, alphaOut, tau0, alphaStar, loglike
    # return loglike


if __name__ == "__main__":
    parser = ArgumentParser(description="")

    parser.add_argument("dirname", help="Path to data directory")
    parser.add_argument("--outfn", type=str, default='', help="Path to data directory")
    parser.add_argument("--initial", type=int, help="Initial index to run")
    parser.add_argument("--final", type=int, help="Final index to run")
    parser.add_argument("--save", action='store_true', help="Whether to save output to file")
    parser.add_argument("--reload", action='store_true', help="Whether to reload data")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute probs and uvlf")
    parser.add_argument("--plot", action='store_true', help="Whether to create diagnostic plots")
    parser.add_argument("--like", action='store_true', help="Whether to calculate and return the likelihood")
    args = parser.parse_args()

    Vouts = []
    alphaOuts = []
    alphaStars = []
    tau0s = []
    loglikes = []
    # results = np.zeros((args.final+1-args.initial,2))
    for i in range(args.initial, args.final+1):
        # results[i-args.initial,:] = [i, run(args.dirname, i, args.task, args.tag, args.reload, args.like)]
        Vout, alphaOut, tau0, alphaStar, loglike = run(args.dirname, i, args.reload, args.recompute, args.plot, args.like)
        # loglike = run(args.dirname, i, args.reload, args.recompute, args.plot, args.like)
        Vouts.append(Vout)
        alphaOuts.append(alphaOut)
        alphaStars.append(alphaStar)
        tau0s.append(tau0)
        loglikes.append(loglike)

            
    if args.save:
        # np.save(args.dirname+'_results.npy',results)
        idxs = list(range(args.initial, args.final+1))
        data = np.array([Vouts, alphaOuts, tau0s, alphaStars, loglikes]).T
        df = pd.DataFrame(data, columns=['velocityOutflow', 'alphaOutflow', 'timescale', 'alphaStar', 'loglike'], index=idxs)
        if len(args.outfn)>0:
            df.to_csv(f'{args.outfn}.csv')
        else:
            df.to_csv(f'{args.dirname}.csv')

