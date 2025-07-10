import numpy as np
import os.path as path
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import yaml 
import pandas as pd
import seaborn as sns
from collections import namedtuple
import analysis
import cmasher as cmr

# fields = ['absolute_mean', 'absolute_sigma', 'absolute_min', 'absolute_max', 
#         'apparent_mean', 'apparent_sigma', 'apparent_min', 'apparent_max']
# Stats = namedtuple('Stats', fields)

# plot aesthetics, change as you see fit
mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['figure.titlesize'] = 23
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 6
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'

from numpy.typing import NDArray


def _get_confidence_intervals_bounds(
    magnitude_grid: NDArray, 
    probs: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """ Gets confidence intervals bounds from pdf array for plotting. 

    This is a helper function specifically for plotting purposes.
    
    Args:
        magnitude_grid: Magnitude grid.
        probs: Array of probabilities.
    Returns:
        Tuple of peak, one sigma upper, two sigma upper, one sigma lower,
        and two sigma lower bounds."""
    nm = probs.shape[1]
    peak = []
    one_sigma_upper = []
    two_sigma_upper = []
    one_sigma_lower = []
    two_sigma_lower = []
    for j in range(nm):
        p = probs[:,j]
        pj = np.argmax(p)
        pmax = np.amax(p)
        p/=pmax
        peak.append(magnitude_grid[pj])

        upper = p[pj+1:]
        upper = upper[np.nonzero(upper)]
        if len(upper)==0: 
            one_sigma_upper.append(magnitude_grid[pj])
            two_sigma_upper.append(magnitude_grid[pj])
        else:
            osui = np.argmin(np.abs(upper-np.exp(-0.5)))
            one_sigma_upper.append(magnitude_grid[osui+pj])
            tsui = np.argmin(np.abs(upper-np.exp(-2.0)))
            two_sigma_upper.append(magnitude_grid[tsui+pj])

        lower = p[:pj] 
        iszero = np.isclose(lower, 0)
        nz = np.sum(iszero)
        lower = lower[np.logical_not(iszero)]
        if len(lower)==0:
            one_sigma_lower.append(magnitude_grid[pj])
            two_sigma_lower.append(magnitude_grid[pj])
        else:
            osli = np.argmin(np.abs(lower-np.exp(-0.5)))
            one_sigma_lower.append(magnitude_grid[nz+osli])
            tsli = np.argmin(np.abs(lower-np.exp(-2.0)))
            two_sigma_lower.append(magnitude_grid[nz+tsli])
    peak = np.array(peak)
    return peak, one_sigma_upper, two_sigma_upper, one_sigma_lower, two_sigma_lower


def plot_probs(
    data: pd.DataFrame, 
    stats: analysis.Stats,
    data_directory: str, 
    do_abs: bool,
    do_skewed: bool,
):
    """ Function to plot the data and pdf of the fit to the g-h connection.

    Args:
        data_log_halo_mass: Array of log halo masses.
        data_magnitudes: Array of magnitudes.
        redshift: Array of redshifts.
        means: Array of mean values.
        sigmas: Array of standard deviation values.
        mins: Array of minimum magnitude per halo mass.
        maxs: Array of maximum magnitude per halo mass.
        data_directory: Directory to save data.
        output_tag: Tag for output files.
        do_abs: Flag to determine absolute magnitude.
    """
    # simple adjustments can be made to show multiple redshifts
    plot_redshifts = [12.0]
    f, ax = plt.subplots(1, len(plot_redshifts), figsize=(5,5),constrained_layout=True, sharey=True)
    colors = sns.color_palette("Blues",n_colors=2)

    # note: mag grid needs to be much wider than plot range for norm. purposes
    if do_abs:
        magnitude_grid = np.linspace(-25.0, 0.0, 1000)
    else:
        magnitude_grid = np.linspace(22.0, 45.0, 1000)

    # needs to gets pdf for a much finer grid in magnitude for plotting purposes
    if do_skewed:
        probs = analysis.get_skewed_probs(magnitude_grid, stats, data_directory, do_abs, True, False)
    else:
        probs = analysis.get_probs(magnitude_grid, stats, data_directory, do_abs, True, False)
    for i,z in enumerate(plot_redshifts):
        idx = data['redshift']==z
        data_log_halo_masses_z = data['log_halo_mass'][idx]
        if do_abs:
            data_magnitudes_z = data['absolute_magnitude'][idx]
        else:
            data_magnitudes_z = data['apparent_magnitude'][idx]
        
        ax.scatter(data_log_halo_masses_z, data_magnitudes_z, 
                   marker='x', c='k', label="Sim. data")

        # extracting CIs from the pdf
        prob_z_index = analysis.redshift_grid == z
        probs_z = probs[:,prob_z_index,:].reshape(len(magnitude_grid), analysis.n_mass_bins)
        ci_bounds_output = _get_confidence_intervals_bounds(magnitude_grid, probs_z)
        peak = ci_bounds_output[0] 
        one_sigma_upper = ci_bounds_output[1] 
        two_sigma_upper = ci_bounds_output[2] 
        one_sigma_lower = ci_bounds_output[3]
        two_sigma_lower = ci_bounds_output[4]

        # marking the 5 sigma depths of the surveys
        reds = sns.color_palette("Reds",n_colors=2)
        ngdeep_five_sigma_depth = 30.4
        ceers_five_sigma_depth = 29.15
        ax.hlines(ngdeep_five_sigma_depth, 8.0,  11.5, linestyle='--', 
                  color=reds[1], label='NGDEEP $5\sigma$ depth', zorder=8)
        ax.arrow(11.1, ngdeep_five_sigma_depth, 0, -0.2, color=reds[1], 
                 linewidth=1.5, head_width=0.03, head_length=0.05)
        ax.hlines(ceers_five_sigma_depth, 8.0, 11.5, linestyle='--', 
                  color=reds[0], label='CEERS $5\sigma$ depth', zorder=7)
        ax.arrow(11.1, ceers_five_sigma_depth, 0, -0.2, color=reds[0], 
                 linewidth=1.5, head_width=0.03, head_length=0.05)

        # CIs
        ax.fill_between(analysis.bin_centers, two_sigma_lower, two_sigma_upper, alpha=0.65, color=colors[0], zorder=5)
        ax.fill_between(analysis.bin_centers, one_sigma_lower, one_sigma_upper, alpha=0.85, color=colors[1], zorder=6)

        # title, legend, labels, etc.
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
        # ax.set_xlim(8.0, 11.5)
        ax.legend(frameon=False, fontsize=12)
    ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\,/\,M_{\odot}\right)$')
    if do_abs:
        # plt.savefig(path.join(data_directory, 'skewed_absolute_sim_data_and_fit.pdf'))
        # plt.savefig('skewed_absolute_sim_data_and_fit.pdf')
        plt.savefig('absolute_sim_data_and_fit.pdf')
    else:
        # plt.savefig(path.join(data_directory, 'skewed_apparent_sim_data_and_fit.pdf'))
        # plt.savefig('skewed_apparent_sim_data_and_fit.pdf')
        plt.savefig('apparent_sim_data_and_fit.pdf')
    plt.close('all')


def _average_uvlf(
    redshifts: NDArray, 
    lower_bound: float, 
    upper_bound: float, 
    uvlf: NDArray, 
    magnitude_grid: NDArray, 
    volumes: NDArray, 
    app_cutoff: float,
):
    """ Averages the UVLF over a redshift range.

    Applies a cutoff during the average based on 5 sigma limiting depth of the
    survey. 
    """
    dz = redshifts[1]-redshifts[0]
    # do_old = True
    z_left = redshifts-dz/2.0
    z_right = redshifts+dz/2.0
    # if do_old:
        
    zidx = (z_left >= lower_bound) & (z_right <= upper_bound)
        # print(lower_bound, upper_bound, z_left[zidx], z_right[zidx])
    # else:
    #     zidx = (redshifts >= lower_bound) & (redshifts <= upper_bound)
        # print(lower_bound, upper_bound, z_left[zidx], z_right[zidx])

    redshift_slice = redshifts[zidx]
    z_left_slice = z_left[zidx]
    z_right_slice = z_right[zidx]
    zuvlf = uvlf[:,zidx]
    zvolume = volumes[zidx]
    totalV = np.zeros_like(magnitude_grid)
    averaged_uvlf = np.zeros_like(magnitude_grid)
    for i,muv in enumerate(magnitude_grid):
        for j in range(len(redshift_slice)): 
            right_cutoff = (app_cutoff - analysis.cosmo.distmod(z_right_slice[j]).value 
                            + 2.5*np.log10(1+z_right_slice[j]))
            if muv < right_cutoff:
                averaged_uvlf[i] += zuvlf[i,j]*zvolume[j]
                totalV[i] += zvolume[j]
    averaged_uvlf /= totalV
    return averaged_uvlf

def _plot_survey_uvlf_data(ax1: plt.Axes, ax2: plt.Axes):
    """ Adds the survey data to the UVLF plot.
    
    Args:
        ax1: plt.Axes for the z~9 plot.
        ax2: plt.Axes for the z~11 plot.
    """
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

    ax1.scatter(ngdeep_muv, ngdeep_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',  label='NGDEEP (Leung et. al. 2023)')
    # ax.errorbar(ngdeep_muv, obs_log_phi, elinewidth=1.8, yerr=ngdeep_log_err, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax1.errorbar(ngdeep_muv, ngdeep_log_phi, yerr=ngdeep_log_err, 
                elinewidth=1.8,  marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax1.errorbar([-21.1],[np.log10(8.9e-5)], yerr=ngdeep_log_err[0][0], uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax1.scatter(ceers_muv_z9, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    # ax.errorbar(ceers_muv_z9, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax1.errorbar(ceers_muv_z9, ceers_log_phi, yerr=ceers_log_err,
                marker='o', color="none", ecolor='gray', capsize=3.5, ls='none',elinewidth=1.8) #  
    # ax1.errorbar(ceers_muv_z9[0], ceers_log_phi[0], yerr=[[ceers_log_err[0][0]],[ceers_log_upper_err2[0]]], #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8) # ls='none'
    # ax1.errorbar([-22.5, ceers_muv_z9[0],-21.5], [np.log10(0.9e-5),ceers_log_phi[0],np.log10(0.9e-5)], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 
    ax1.errorbar([-22.5, -21.5], [np.log10(0.9e-5),np.log10(0.9e-5)], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 
    
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

    ax2.scatter(ngdeep_muv, obs_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k',label='NGDEEP data (Leung et. al. 2023)',)
    # ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax2.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    ax2.errorbar([-20.05],[np.log10(9.7e-5)], yerr=log_err[0][0], uplims=True,
                elinewidth=1.8, color="none", ecolor='k', ls='none', capsize=3.5)
    ax2.scatter(ceers_muv_z11, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    # ax.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax2.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    # ax2.errorbar([ceers_muv_z11[0],ceers_muv_z11[-1]], [ceers_log_phi[0],ceers_log_phi[-1]], yerr=[[ceers_log_err[0][0],ceers_log_err[0][-1]],[ceers_log_err2[1][0],ceers_log_err2[1][-1]]], #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') # ls='none'
    ax2.errorbar([-21.0], [np.log10(0.5e-5)], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
                ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 
    # ax2.errorbar([-21.0, -20.5,ceers_muv_z11[-1]], [np.log10(0.5e-5),ceers_log_phi[0],ceers_log_phi[-1]], yerr=ceers_log_err[0][0], uplims=True, #uplims=[True],
    #             ecolor='gray',color="gray", capsize=3.5, elinewidth=1.8, ls='None') 


def plot_uvlf(
    magnitude_grid: NDArray, 
    redshifts: NDArray, 
    uvlf: NDArray, 
    volumes: NDArray, 
    data_directory: str,
    axs: plt.Axes=None,
    c: str='k', # its not only a string, don't know how to type hint this
    save: bool=True,
):
    """ Plots the uvlf with survey data overlaid. 
    """
    # print("plot uvlf")
    app_cutoff = 30.4
    if axs is None:
        f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)
    _plot_survey_uvlf_data(axs[0],axs[1])

    ax = axs[0]
    averaged_uvlf = _average_uvlf(redshifts, 8.5, 9.5, uvlf, 
        magnitude_grid, volumes, app_cutoff)
    non_zero_idx = averaged_uvlf > 0
    averaged_uvlf = np.log10(averaged_uvlf[non_zero_idx])
    ax.plot(magnitude_grid[non_zero_idx], averaged_uvlf, label='Sim.', lw=2.5, c=c)

    ax.set_title(r'$8.5<z<9.5$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_xlim(-23.0, -16.5)
    ax.set_ylim(-7, -1.5)
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    
    # z~11 plot
    ax = axs[1]
    averaged_uvlf = _average_uvlf(redshifts, 9.5, 12.0, uvlf,
        magnitude_grid, volumes, app_cutoff)
    non_zero_idx = averaged_uvlf > 0
    averaged_uvlf = np.log10(averaged_uvlf[non_zero_idx])
    ax.plot(magnitude_grid[non_zero_idx], averaged_uvlf, label='Sim.', lw=2.5, c=c)

    ax.set_title(r'$9.5<z<12.0$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(-23.0, -16.5)
    ax.set_ylim(-7, -1.5)
    if save:
        plt.savefig(path.join(data_directory, 'uvlf.pdf'))
        plt.close('all')


def plot_sampled_uvlf(file_base, df, bestfit_uvlf):
    app_cutoff = 30.4
    f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True, sharey=True)


    bestfit_z9 = _average_uvlf(analysis.redshift_grid, 8.5, 9.5, bestfit_uvlf, 
            analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)
    bestfit_z11 = _average_uvlf(analysis.redshift_grid, 9.5, 12.0, bestfit_uvlf, 
            analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)
    
    # new_bestfit_z9 = _average_uvlf(analysis.redshift_grid, 8.5, 9.5, bestfit_uvlf, 
    #         analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)
    # new_bestfit_z11 = _average_uvlf(analysis.redshift_grid, 9.5, 12.0, bestfit_uvlf, 
    #         analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)

    plot_uvlf(
        analysis.absolute_magnitude_grid, 
        analysis.redshift_grid, 
        bestfit_uvlf/analysis.dabs, 
        analysis.z_volumes, 
        bestfit_directory,
        axs = axs,
        save = False,
    )

    n_sample = 1000 # 5000
    colors = sns.color_palette("Blues",n_colors=2) 

    z9_upper = np.zeros_like(analysis.absolute_magnitude_grid)
    z9_lower = np.ones_like(analysis.absolute_magnitude_grid)
    z11_upper = np.zeros_like(analysis.absolute_magnitude_grid)
    z11_lower = np.ones_like(analysis.absolute_magnitude_grid)
    samples = df.sample(n_sample, replace=True, weights='like')
    z9_ys = []
    z11_ys = []
    # print()
    for index,row in samples.iterrows():
        # not sure why but some idx get saved as floats?
        fn = file_base+f'_p{int(round(row["idx"]))}/'
        uvlf = analysis.get_uvlf(None, None, fn, True, False, False)
        uvlf /= analysis.dabs
        z9_uvlf = _average_uvlf(analysis.redshift_grid, 8.5, 9.5, uvlf, 
            analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)
        z9_ys.append(z9_uvlf)
        
        z11_uvlf = _average_uvlf(analysis.redshift_grid, 9.5, 12.0, uvlf, 
            analysis.absolute_magnitude_grid, analysis.z_volumes, app_cutoff)  
        z11_ys.append(z11_uvlf)

    z9_ys = np.array(z9_ys)
    z11_ys = np.array(z11_ys)

    z9_diff = np.abs(z9_ys-bestfit_z9) 
    z11_diff = np.abs(z11_ys-bestfit_z11) 

    for i,frac in enumerate([0.68, 0.95]):
        z11_lower = np.zeros_like(analysis.absolute_magnitude_grid)
        z9_lower = np.zeros_like(analysis.absolute_magnitude_grid)
        z11_upper = np.zeros_like(analysis.absolute_magnitude_grid)
        z9_upper = np.zeros_like(analysis.absolute_magnitude_grid)
        c = colors[-(i+1)]
        n_cut = int(round(frac*n_sample))
        for im in range(len(analysis.absolute_magnitude_grid)):
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
        ax.fill_between(analysis.absolute_magnitude_grid[idx], z9_lower, z9_upper, color=c, alpha=0.9-i*0.2, zorder=-i, 
                        label=f'${int(round(frac*100))}'+r'\% $ CI')
        idx1 = idx
        
        idx = z11_lower > 0
        z11_upper = np.log10(z11_upper[idx])
        z11_lower = np.log10(z11_lower[idx])
        ax = axs[1]
        ax.fill_between(analysis.absolute_magnitude_grid[idx], z11_lower, z11_upper, color=c, alpha=0.9-i*0.2, zorder=-i, 
                        label=f'${int(round(frac*100))}'+r'\% $ CI')

        if i == 0:
            oneSigmaScatter = [[analysis.absolute_magnitude_grid[idx1],z9_upper-z9_lower], [analysis.absolute_magnitude_grid[idx], z11_upper-z11_lower]]
        if i == 1:
            twoSigmaScatter = [[analysis.absolute_magnitude_grid[idx1],z9_upper-z9_lower], [analysis.absolute_magnitude_grid[idx], z11_upper-z11_lower]]
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
    ax.legend(frameon=False, fontsize=12.5, loc='upper left')
    ax.set_xlim(-22.75, -17.1)
    ax.set_ylim(-6.0, -2.0)
    ax.set_yticks([-6,-5,-4,-3,-2])

    plt.savefig(f'sampled_uvlf_CI.pdf')
    plt.close('all')


def plot_Mh_given_fixed_Muv(data_directory, abs_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""

    ## just doing fixed z for illustration ##
    # f, axs = plt.subplots(3, 1, figsize=(6,15), constrained_layout=True)
    z = 12.0
    zidx = analysis.redshift_grid==z
    f, ax = plt.subplots(1, 1, figsize=(6,5), constrained_layout=True)
    # P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
    # print(np.sum(zidx))
    P_mh = bweights / np.sum(bweights)
    prob_mh_given_Muv = abs_probs[:,zidx,:] *  P_mh[zidx,:]
    prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
    # print(prob_mh_given_Muv.shape)
    # print(prob_mh_given_Muv.shape)
    # for i,z in enumerate([8.0,12.0,16.0]):
    # for i,z in enumerate([12.0]):
    # zidx = analysis.redshift_grid==z
    for j,M_uv in enumerate([-17, -19, -21]):
        c = plt.cm.Dark2(j)
        midx = analysis.absolute_magnitude_grid==M_uv
        prob = prob_mh_given_Muv[midx]
        ax.plot(analysis.bin_centers, prob.flatten(), c=c, label=r'$M_{\mathrm{uv}}='+str(M_uv)+'$')
    ax.set_ylabel(r'$P(M_{h}|M_{\mathrm{uv}})$')
    ax.set_ylim(0,0.25)
    ax.set_xlim(8.5, 11.25)
    ax.set_title(f'$z={z}$',fontsize=20)
    ax.legend(frameon=False, fontsize=13)
    ax.set_xlabel(r'$M_{h}$')

    # plt.savefig(path.join(data_directory, 'prob_mh_given_muv.pdf'))
    plt.savefig('prob_mh_given_fixed_muv.pdf')
    plt.close('all')
    

def plot_Mh_from_data(data_directory, app_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""
    # f, axs = plt.subplots(3, 1, figsize=(6,15), constrained_layout=True)
    zidx = analysis.redshift_grid >= 8.5
    P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
    prob_mh_given_Muv = app_probs *  P_mh
    prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
    ng, nmuv, nz = analysis.ngdeep_pdf.shape
    # print(ng, nmuv, nz)
    # print(prob_mh_given_Muv.shape)
    ngd_pdf = analysis.ngdeep_pdf.reshape(ng,nmuv,nz,1)
    c_pdf = analysis.ceers_pdf.reshape(-1,nmuv,nz,1)
    prob_mh_given_Muv = prob_mh_given_Muv.reshape(1, nmuv, nz, -1)
    
    prob_mh = np.sum(ngd_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
    prob_mh += np.sum(c_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
    prob_mh /= np.sum(prob_mh)
    plt.plot(analysis.bin_centers, prob_mh)
    plt.ylabel(r'$P(M_{h}|\mathcal{D})$')
    # plt.ylim(0,0.25)
    plt.xlim(8.5, 11.25)
    # axs[i].set_title(f'$z={z}$')
    plt.xlabel(r'$M_{h}$')
    plt.tight_layout()
    plt.savefig(path.join(data_directory, 'prob_mh_given_data.pdf'))
    plt.close('all')


def plot_Mh_from_data_z(data_directory, app_probs, bweights):
    """Abs_probs is P(Muv|Mh) N_uv, N_z, N_h"""
    # f, axs = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True, sharey=True)
    interp_mh = np.linspace(8.0, 11.5, 1000)

    for i in range(3):
        i = 2-i
        # ax = axs[i]
        if i==0:
            zidx = (analysis.redshift_grid >= 8.5) & (analysis.redshift_grid<9.5)
        elif i==1:
            zidx = (analysis.redshift_grid >= 9.5) & (analysis.redshift_grid<12.0)
        else:
            zidx = (analysis.redshift_grid >= 8.5)

        P_mh = bweights / np.sum(bweights, axis=1).reshape(-1,1) # bweights should be z by Mh
        prob_mh_given_Muv = app_probs *  P_mh
        prob_mh_given_Muv /= np.sum(prob_mh_given_Muv, axis=0)
        ng, nmuv, nz = analysis.ngdeep_pdf.shape
        # print(ng, nmuv, nz)
        # print(prob_mh_given_Muv.shape)
        ngd_pdf = analysis.ngdeep_pdf.reshape(ng,nmuv,nz,1)
        c_pdf = analysis.ceers_pdf.reshape(-1,nmuv,nz,1)
        prob_mh_given_Muv = prob_mh_given_Muv.reshape(1, nmuv, nz, -1)
        
        prob_mh = np.sum(ngd_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
        prob_mh += np.sum(c_pdf[:,:,zidx,:]*prob_mh_given_Muv[:,:,zidx,:],axis=(0,1,2))
        prob_mh /= np.max(prob_mh) # np.sum(prob_mh)

        prob_mh = np.interp(interp_mh, analysis.bin_centers, prob_mh)

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
    plt.xlabel(r'$\mathrm{Log}(M_{h}\, \left[M_{\odot}\right])$')
    plt.ylabel(r'$P(M_{h}|\mathcal{D})$')
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=12)
    # plt.savefig(path.join(data_directory, 'prob_mh_given_data_z.pdf'))
    plt.savefig('prob_mh_given_data_z.pdf')
    plt.close('all')


def parameters_to_labels(parameters: list) -> list:
    """ Returns list of latex labels for corresponding parameter strings. """
    labels = []
    for p in parameters:
        if (p == 'outflow_velocity') or (p=='velocityOutflow'):
            tex = r'$V_{\mathrm{outflow}}\,\mathrm{[km/s]}$'
        elif p == 'outflow_alpha' or (p=='alphaOutflow'):
            tex = r'$\alpha_{\mathrm{outflow}}$'
        elif p == 'sfr_timescale' or (p=='timescale'):
            tex = r'$\tau_{0}\,\mathrm{[Gyr]}$'
        elif p == 'sfr_alpha' or (p=='alphaStar'):
            tex = r'$\alpha_{\ast}$'
        else:
            raise Exception('Unknown Astro Parameter')
        labels.append(tex)
    return labels


def calculate_1d_posterior(df: pd.DataFrame, p1: str) -> NDArray:
    """
    Assuming all parameters are uniformly distributed (as is the case for our
    sampling in https://arxiv.org/abs/2410.11680).

    Let's just do this two parameters at a time.
    """
    parameter1_unique_values = df[p1].unique()
    parameter1_unique_values.sort()
    like = df['like'].values
    marginalized_posterior = np.zeros(len(parameter1_unique_values))
    for i, val1 in enumerate(parameter1_unique_values):
        idx = np.isclose(df[p1].to_numpy(), val1) 
        # assuming uniform priors, we can just sum the likelihood values
        marginalized_posterior[i] = np.sum(like[idx]) 
    norm = np.amax(marginalized_posterior)
    return marginalized_posterior / norm


def calculate_2d_posterior(df: pd.DataFrame, p1: str, p2: str) -> NDArray:
    """
    Assuming all parameters are uniformly distributed (as is the case for our
    sampling in https://arxiv.org/abs/2410.11680).

    Let's just do this two parameters at a time.
    """
    parameter1_unique_values = df[p1].unique()
    parameter1_unique_values.sort()
    parameter2_unique_values = df[p2].unique()
    parameter2_unique_values.sort()
    like = df['like'].values
    marginalized_posterior = np.zeros((len(parameter1_unique_values), 
                                       len(parameter2_unique_values)))
    for i, val1 in enumerate(parameter1_unique_values):
        for j, val2 in enumerate(parameter2_unique_values):
            idx = np.isclose(df[p1].to_numpy(), val1) & np.isclose(df[p2].to_numpy(), val2)
            # assuming uniform priors, we can just sum the likelihood values
            marginalized_posterior[i,j] = np.sum(like[idx])
    norm = np.amax(marginalized_posterior)
    return marginalized_posterior / norm


def label_axis(axs, i, j, labels, uniques, parameters):
    ax = axs[i,j]
    pi = parameters[i]
    pj = parameters[j]
    nparam = len(parameters)
    if (j==0) and (i!=0):
        # ax.get_shared_y_axes().join(ax, *axs[i,:i])
        # for ax2 in axs[i,:i]:
            # ax.sharey(ax2)
        ax.set_ylabel(labels[i])
    elif j<i:
        ax.set_yticklabels([])
        ax.tick_params(left=True, right=True)
    if i==(nparam-1):
        ax.set_xlabel(labels[j])


def get_flat_data_points(p,data):
    new_data = np.zeros(len(data)+1)
    delta = (data[1]-data[0])/2.0
    new_data[0] = data[0]-delta
    new_data[1:] = data+delta
    return new_data


def plot_1d(i, ax, values, tab_prob_1d, label):
    """ Plots the 1d marginalized posterior for parameter i. 
    
    Includes shaded 68% and 95% confidence intervals. Since the distributions 
    are skewed, this is done by handing, adding up the probabilities until 
    symmetrically around the peak of the distribution until the desired area 
    under the curve is attained. """
    originalProb = np.array(tab_prob_1d)
    # prob /= np.sum(prob)
    interp_values = np.linspace(np.amin(values), np.amax(values), (len(values)-1)*10+1)
    prob = np.interp(interp_values, values, originalProb)
    prob /= np.sum(prob)
    idx = np.argmax(prob)
    left = max(0, idx - 1)
    right = idx + 1
    area = np.sum(prob[left:right])
    length = len(prob)
    while area < 0.68:
        # print(leftProb, rightProb)
        leftProb = prob[left]
        rightProb = prob[right]
        if (left == 0) and (right == (length-1)):
            # print("summing entire array")
            break
        elif (left == 0):
            # print("end of left")
            right += 1 
        elif (right == (length-1)):
            # print("end of right")
            left -= 1
        elif leftProb < rightProb:
            right += 1
        else: 
            left -= 1   
        area = np.sum(prob[left:right])
        
    left1 = left
    right1 = right
    # left = one
    leftOneSigma = interp_values[left]
    rightOneSigma = interp_values[right]
    print(f'68% confindence interval is {leftOneSigma}-{rightOneSigma}')
    print(f'Actual area was {area}')

    left = max(0, idx - 1)
    right = idx + 1
    area = np.sum(prob[left:right])
    while area < 0.95:
        leftProb = prob[left]
        rightProb = prob[right]
        if (left == 0) and (right == (length-1)):
            # print("summing entire array")
            break
        elif (left == 0):
            # print("end of left")
            right += 1 
        elif (right == (length-1)):
            # print("end of right")
            left -= 1
        elif leftProb < rightProb:
            right += 1
        else: 
            left -= 1   
        area = np.sum(prob[left:right])
        
    # print(leftProb, rightProb)
    left2 = left
    right2 = right
    leftTwoSigma = interp_values[left]
    rightTwoSigma = interp_values[right]

    print(f'95% confindence interval is {leftTwoSigma}-{rightTwoSigma}')
    print(f'Actual area was {area}')

    # plt.plot(vouts, originalProb, 'k')
    ax.plot(interp_values, prob, 'k')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.fill_between(interp_values[left1:right1], [0]*(right1-left1), prob[left1:right1], color='k', 
                    zorder=-998, alpha=0.6)
    ax.fill_between(interp_values[left2:right2], [0]*(right2-left2), prob[left2:right2], color='k', 
                    zorder=-999, alpha=0.3)


def plot_astro_like(df, parameters):
    labels = parameters_to_labels(parameters)
    # df['like'] = like
    maxidx = df['like'].idxmax()
    print(maxidx)
    nparam = len(parameters)
    uniques = [np.sort(np.unique(df[p])) for p in parameters]
    # print(uniques)
    print('Max likehood is {}'.format(df['like'][maxidx]))
    print('MLE parameters are {},{},{},{}'.format(*[df[p][maxidx] for p in parameters]))

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cmr.get_sub_cmap('cmr.neutral_r', 0.05, 1.0)
    frac = 0.10
    f,axs = plt.subplots(nparam, nparam, figsize=(11*(1+frac),11), sharex='col')
    for i,pi in enumerate(parameters):
        for j,pj in enumerate(parameters):
            ax = axs[i,j]
            label_axis(axs,i,j,labels,uniques,parameters)
            if j<(i+1):
                if i==j:
                    xs = uniques[i]
                    posterior = calculate_1d_posterior(df, pi)
                    plot_1d(i, ax, xs, posterior, labels[i])
                if i!=j:
                    # print(pi,pj)
                    prob = calculate_2d_posterior(df, pi, pj)
                    # print(prob)
                    x = get_flat_data_points(pj, uniques[j])
                    y = get_flat_data_points(pi, uniques[i])
                    xx,yy = np.meshgrid(x,y)
                    ax.pcolormesh(xx, yy, prob, 
                                        cmap = cmap, norm = norm, shading = 'flat')                
            else:
                ax.axis('off')
    # plt.minorticks_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pad = 0.01
    cbar = f.colorbar(sm, ax=axs[-1,:], pad=pad, #extend='min',
                      fraction=frac)
    f.colorbar(sm, ax=axs[0,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[1,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[2,:], pad=pad, fraction=frac).ax.set_visible(False)
    plt.savefig('triangle_like.pdf', bbox_inches='tight')
    # plt.savefig('skewed_triangle_like.pdf', bbox_inches='tight')
    plt.close('all')

do_skewed = False

df = pd.read_csv('paper_params.csv')
df = df.sort_values('loglike', ascending=False)
df.insert(len(df.columns), 'like', np.exp(df['loglike']))
df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
print(df.head(n=10))



parameters = ['outflow_velocity', 'outflow_alpha', 'sfr_timescale', 
                'sfr_alpha']

base = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data'
dirname = 'paper_params' 
bestfit_index = df['idx'][df['like'].idxmax()]
bestfit_directory = path.join(base, dirname+f'_p{bestfit_index}/')
# print(bestfit_directory)
bestfit_data = analysis.load_data(bestfit_directory, True, False)
bestfit_stats = analysis.get_stats(bestfit_data)
bestfit_abs_probs = analysis.get_probs(analysis.absolute_magnitude_grid, bestfit_stats, 
                                    bestfit_directory, True, False, False)
bestfit_app_probs = analysis.get_probs(analysis.apparent_magnitude_grid, bestfit_stats, 
                                    bestfit_directory, False, False, False)
bestfit_uvlf = analysis.get_uvlf(bestfit_abs_probs, analysis.binned_weights, bestfit_directory, True, False, False)
plot_probs(
    bestfit_data, 
    bestfit_stats,
    bestfit_directory, 
    True,
    do_skewed,
)
plot_probs(
    bestfit_data, 
    bestfit_stats,
    bestfit_directory, 
    False,
    do_skewed,
)
plot_uvlf(
    analysis.absolute_magnitude_grid, 
    analysis.redshift_grid, 
    bestfit_uvlf/analysis.dabs, 
    analysis.z_volumes, 
    bestfit_directory
)
plot_Mh_given_fixed_Muv(bestfit_directory, bestfit_abs_probs, analysis.binned_weights)
plot_Mh_from_data_z(bestfit_directory, bestfit_app_probs, analysis.binned_weights)

file_base = path.join(base,dirname)
plot_sampled_uvlf(file_base, df, bestfit_uvlf)
plot_astro_like(df, parameters)

