"""
Plotting module for JWST analysis results.

This module provides functions to generate publication-quality figures including:
- Galaxy-halo connection probability distributions
- UV luminosity functions with observational data
- Parameter posterior distributions (corner plots)
- Halo mass probability distributions
- Confidence intervals and uncertainty visualizations
"""

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
    """Get confidence interval bounds from probability distribution array.

    Helper function for plotting that extracts 1-sigma and 2-sigma confidence
    intervals from probability distributions. Finds the peak and determines
    upper/lower bounds where the probability drops to exp(-0.5) and exp(-2.0)
    relative to the peak.

    Parameters
    ----------
    magnitude_grid : NDArray
        Grid of magnitudes on which probabilities are evaluated.
    probs : NDArray
        Array of shape (n_mag, n_mass_bins) containing probability distributions.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
        Tuple containing (in order):
        - peak: Peak magnitude for each mass bin
        - one_sigma_upper: Upper 1-sigma bound
        - two_sigma_upper: Upper 2-sigma bound
        - one_sigma_lower: Lower 1-sigma bound
        - two_sigma_lower: Lower 2-sigma bound
    """
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
) -> None:
    """Plot galaxy-halo connection data and fitted probability distributions.

    Creates scatter plots of simulated galaxy data overlaid with confidence
    intervals (1-sigma and 2-sigma) from the fitted galaxy-halo connection
    model. Also marks the 5-sigma limiting depths of NGDEEP and CEERS surveys.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing simulated galaxy data with columns: 'redshift',
        'log_halo_mass', 'absolute_magnitude', 'apparent_magnitude'.
    stats : analysis.Stats
        Statistics object containing fitted parameters for the galaxy-halo
        connection.
    data_directory : str
        Directory containing the analysis results (used to load PDFs).
    do_abs : bool
        If True, plot absolute magnitudes; if False, plot apparent magnitudes.
    do_skewed : bool
        If True, use skewed probability distributions; if False, use normal
        distributions.

    Returns
    -------
    None
        Saves figure to 'absolute_sim_data_and_fit.pdf' or
        'apparent_sim_data_and_fit.pdf' in the current directory.
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
) -> NDArray:
    """Average UV luminosity function over a redshift range.

    Computes volume-weighted average of the UVLF over a specified redshift
    range. Applies a magnitude cutoff based on the 5-sigma limiting depth of
    the survey, accounting for redshift-dependent distance modulus and
    k-correction.

    Parameters
    ----------
    redshifts : NDArray
        Array of redshift values.
    lower_bound : float
        Lower redshift bound for averaging.
    upper_bound : float
        Upper redshift bound for averaging.
    uvlf : NDArray
        UV luminosity function array, shape (n_mag, n_z).
    magnitude_grid : NDArray
        Grid of absolute magnitudes.
    volumes : NDArray
        Comoving volumes for each redshift bin.
    app_cutoff : float
        Apparent magnitude cutoff (5-sigma limiting depth).

    Returns
    -------
    NDArray
        Volume-weighted average UVLF over the specified redshift range,
        shape (n_mag,).
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

def _plot_survey_uvlf_data(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """Add observational survey data to UVLF plots.
    
    Plots NGDEEP and CEERS survey data points with error bars on the UVLF
    plots. Includes both 1-sigma and 2-sigma error bars, and upper limits
    where applicable.

    Parameters
    ----------
    ax1 : plt.Axes
        Axes object for the z~9 plot panel.
    ax2 : plt.Axes
        Axes object for the z~11 plot panel.

    Returns
    -------
    None
        Modifies the axes objects in place.

    References
    ----------
    .. [1] Leung et al. 2023 (NGDEEP survey)
    .. [2] Finkelstein et al. 2023 (CEERS survey)
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
    c: str='k',
    save: bool=True,
) -> None:
    """Plot UV luminosity function with observational survey data overlaid.

    Creates a two-panel figure showing the UVLF at z~9 (left) and z~11 (right)
    with NGDEEP and CEERS observational data points. The model UVLF is
    averaged over the redshift ranges 8.5<z<9.5 and 9.5<z<12.0.

    Parameters
    ----------
    magnitude_grid : NDArray
        Grid of absolute magnitudes.
    redshifts : NDArray
        Array of redshift values.
    uvlf : NDArray
        UV luminosity function array, shape (n_mag, n_z).
    volumes : NDArray
        Comoving volumes for each redshift bin.
    data_directory : str
        Directory where the figure will be saved.
    axs : plt.Axes, optional
        Pre-existing axes objects. If None, creates new figure and axes.
        Default is None.
    c : str, optional
        Color for the model UVLF line. Default is 'k' (black).
    save : bool, optional
        If True, saves the figure to 'uvlf.pdf' in data_directory.
        Default is True.

    Returns
    -------
    None
        Creates and optionally saves the plot.
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


def plot_sampled_uvlf(file_base: str, df: pd.DataFrame, bestfit_uvlf: NDArray) -> None:
    """Plot UVLF with confidence intervals from parameter sampling.

    Creates UVLF plots showing the best-fit model and confidence intervals
    (68% and 95%) computed from a sample of parameter combinations weighted
    by their likelihoods. The sampling is done with replacement using the
    likelihood values as weights.

    Parameters
    ----------
    file_base : str
        Base path to parameter directories (e.g., 'paper_params').
    df : pd.DataFrame
        DataFrame containing parameter combinations and their likelihoods.
        Must have columns: 'idx', 'like'.
    bestfit_uvlf : NDArray
        UVLF array for the best-fit parameter combination.

    Returns
    -------
    None
        Saves figure to 'sampled_uvlf_CI.pdf' in the current directory.
    """
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


def plot_Mh_given_fixed_Muv(
    data_directory: str, 
    abs_probs: NDArray, 
    bweights: NDArray
) -> None:
    """Plot conditional probability distribution P(M_h | M_UV) for fixed magnitudes.

    Computes and plots the probability distribution of halo mass given a fixed
    UV magnitude using Bayes' theorem: P(M_h|M_UV) ∝ P(M_UV|M_h) × P(M_h).

    Parameters
    ----------
    data_directory : str
        Directory where the figure will be saved.
    abs_probs : NDArray
        Probability array P(M_UV|M_h) with shape (n_mag, n_z, n_mass_bins).
    bweights : NDArray
        Binned merger tree weights (proportional to P(M_h)) with shape
        (n_z, n_mass_bins).

    Returns
    -------
    None
        Saves figure to 'prob_mh_given_fixed_muv.pdf' in the current directory.
    """

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
    

def plot_Mh_from_data(
    data_directory: str, 
    app_probs: NDArray, 
    bweights: NDArray
) -> None:
    """Plot posterior probability distribution P(M_h | data).

    Computes the posterior probability distribution of halo mass given the
    observed galaxy data by marginalizing over redshift and apparent magnitude
    uncertainties. Combines NGDEEP and CEERS survey data.

    Parameters
    ----------
    data_directory : str
        Directory where the figure will be saved.
    app_probs : NDArray
        Probability array P(m_app|M_h) with shape (n_mag, n_z, n_mass_bins).
    bweights : NDArray
        Binned merger tree weights with shape (n_z, n_mass_bins).

    Returns
    -------
    None
        Saves figure to 'prob_mh_given_data.pdf' in data_directory.
    """
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


def plot_Mh_from_data_z(
    data_directory: str, 
    app_probs: NDArray, 
    bweights: NDArray
) -> None:
    """Plot posterior probability distribution P(M_h | data) for different redshift bins.

    Similar to `plot_Mh_from_data`, but shows the posterior distribution
    separately for different redshift ranges (8.5≤z<9.5, 9.5≤z<12.0, and all z)
    to illustrate how the inferred halo mass distribution varies with redshift.

    Parameters
    ----------
    data_directory : str
        Directory where the figure will be saved.
    app_probs : NDArray
        Probability array P(m_app|M_h) with shape (n_mag, n_z, n_mass_bins).
    bweights : NDArray
        Binned merger tree weights with shape (n_z, n_mass_bins).

    Returns
    -------
    None
        Saves figure to 'prob_mh_given_data_z.pdf' in the current directory.
    """
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
    """Convert parameter names to LaTeX labels for plotting.

    Parameters
    ----------
    parameters : list
        List of parameter name strings.

    Returns
    -------
    list
        List of LaTeX-formatted labels for each parameter.

    Raises
    ------
    Exception
        If an unknown parameter name is encountered.
    """
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
    """Calculate 1D marginalized posterior distribution for a parameter.

    Computes the posterior probability distribution for a single parameter by
    marginalizing over all other parameters. Assumes uniform priors, so the
    posterior is proportional to the sum of likelihoods for each parameter value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing parameter values and likelihoods. Must have
        columns matching parameter names and a 'like' column.
    p1 : str
        Name of the parameter to marginalize over.

    Returns
    -------
    NDArray
        Normalized posterior probability distribution (max value = 1).

    References
    ----------
    .. [1] https://arxiv.org/abs/2410.11680
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
    """Calculate 2D marginalized posterior distribution for two parameters.

    Computes the joint posterior probability distribution for two parameters
    by marginalizing over all other parameters. Assumes uniform priors, so the
    posterior is proportional to the sum of likelihoods for each parameter
    combination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing parameter values and likelihoods. Must have
        columns matching parameter names and a 'like' column.
    p1 : str
        Name of the first parameter.
    p2 : str
        Name of the second parameter.

    Returns
    -------
    NDArray
        2D array of normalized posterior probabilities (max value = 1),
        shape (n_unique_p1, n_unique_p2).

    References
    ----------
    .. [1] https://arxiv.org/abs/2410.11680
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


def label_axis(
    axs: np.ndarray, 
    i: int, 
    j: int, 
    labels: list, 
    uniques: list, 
    parameters: list
) -> None:
    """Set axis labels for corner plot panels.

    Helper function for creating corner plots that sets appropriate axis
    labels and tick visibility based on panel position.

    Parameters
    ----------
    axs : np.ndarray
        2D array of axes objects.
    i : int
        Row index of the panel.
    j : int
        Column index of the panel.
    labels : list
        List of LaTeX labels for each parameter.
    uniques : list
        List of unique parameter values for each parameter.
    parameters : list
        List of parameter names.

    Returns
    -------
    None
        Modifies axes objects in place.
    """
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


def get_flat_data_points(p: str, data: NDArray) -> NDArray:
    """Create flat data points for pcolormesh plotting.

    Adds half-bin-width offsets to data points to create proper bin edges
    for pcolormesh, which requires edges rather than centers.

    Parameters
    ----------
    p : str
        Parameter name (unused, kept for compatibility).
    data : NDArray
        Array of bin center values.

    Returns
    -------
    NDArray
        Array of bin edges with one extra element.
    """
    new_data = np.zeros(len(data)+1)
    delta = (data[1]-data[0])/2.0
    new_data[0] = data[0]-delta
    new_data[1:] = data+delta
    return new_data


def plot_1d(
    i: int, 
    ax: plt.Axes, 
    values: NDArray, 
    tab_prob_1d: NDArray, 
    label: str
) -> None:
    """Plot 1D marginalized posterior with confidence intervals.

    Plots the 1D posterior distribution with shaded 68% and 95% confidence
    intervals. Since distributions may be skewed, confidence intervals are
    computed by symmetrically expanding around the peak until the desired
    probability mass is contained.

    Parameters
    ----------
    i : int
        Parameter index (for identification/debugging).
    ax : plt.Axes
        Axes object on which to plot.
    values : NDArray
        Array of parameter values.
    tab_prob_1d : NDArray
        Array of posterior probabilities corresponding to values.
    label : str
        LaTeX label for the parameter (currently unused but kept for
        compatibility).

    Returns
    -------
    None
        Modifies the axes object in place. Prints confidence interval
        information to console.
    """
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


def plot_astro_like(df: pd.DataFrame, parameters: list) -> None:
    """Create corner plot (triangle plot) of parameter posterior distributions.

    Generates a corner plot showing 1D marginalized posteriors on the diagonal
    and 2D joint posteriors in the lower triangle. The upper triangle is left
    blank. Uses a colormap to show posterior probability density.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing parameter values and likelihoods. Must have
        columns matching parameter names and a 'like' column.
    parameters : list
        List of parameter names to include in the corner plot.

    Returns
    -------
    None
        Saves figure to 'triangle_like.pdf' in the current directory.
        Prints maximum likelihood parameter values to console.
    """
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
# df = df.sort_values('loglike', ascending=False)
# df.insert(len(df.columns), 'like', np.exp(df['loglike']))
# df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
# print(df.head(n=10))

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

