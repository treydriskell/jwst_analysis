
import h5py
import numpy as np
import os.path as path
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed
from numpy.typing import NDArray
from collections import namedtuple
import scipy
from time import time
# from scipy.optimize import minimize
# import scipy.stats as stats


# this is just a convenient way to store the stats calculated from the data
fields = ['absolute_mean', 'absolute_sigma', 'absolute_min', 'absolute_max', 
        'apparent_mean', 'apparent_sigma', 'apparent_min', 'apparent_max']
# below was used for skewed distributions which have been abandoned
# fields = ['absolute_median', 'absolute_sigma_left', 'absolute_sigma_right', 'absolute_min', 'absolute_max', 
#         'apparent_median', 'apparent_sigma_left', 'apparent_sigma_right', 'apparent_min', 'apparent_max', ]
Stats = namedtuple('Stats', fields)

rng = np.random.default_rng()

# cosmology to be used throughout the analysis once
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)


def get_weights_from_hmf(
    filename: str,
    load: bool = True,
    save: bool = True,
) -> NDArray:
    """Converts hmf output by Galacticus to corresponding merger tree weights.
    
    This is done to evaluate the hmf at intermediate z. Corresponding 
    merger tree weights will be saved to output file data/hmf_weights.npy if 
    save is True. If load is True, data will be loaded instead of recomputed
    unless unavailable. 
    
    Note: currently the number of redshifts the array is evaluated on is hardcoded (17).

    Parameters:
        filename: path to the (input) hmf hdf5 file. 
        load: if True, will attempt to load file first 
        save: if True, will save the output to file data/hmf_weights.npy 

    Returns:
        A numpy array of the merger tree weights evaluated on a grid of z values
        in increasing order of redshift.
    """
    output_filename = 'data/hmf_weights.npy' 
    if load and path.isfile(output_filename):
        weights = np.load(output_filename) 
    else:
        weights = []
        f = h5py.File(filename,"r")
        halo_masses = np.geomspace(1.0e8, 5.0e11, 3699) # Warning: Hard coded!
        log_halo_masses = np.log10(halo_masses)
        log_delta_mh = log_halo_masses[1] - log_halo_masses[0]
        left_bin_edge = 10**(log_halo_masses - log_delta_mh/2.0)
        right_bin_edge = 10**(log_halo_masses + log_delta_mh/2.0)
        bin_size = right_bin_edge - left_bin_edge
        weights = []
        for i in range(17):
            j = 17-i # Warning: Hard coded!
            output = f[f'Outputs/Output{j}']
            hmf = output['haloMassFunctionM'] 
            weight = bin_size * hmf
            weights.append(weight)
        weights = np.array(weights)
        f.close()
        if save:
            np.save(output_filename, weights)
    return weights


def get_data_from_hdf5(filename: str, jwst_filter_name: str) -> NDArray:
    """ Extracts the log halo masses, magnitudes, and weights from galacticus hdf5 file.

    Args:
        filename: path to hdf5 file. 
        jwst_filter_name: name of jwst filter for luminosities.
    """
    outfile = h5py.File(filename, "r")
    outputs = outfile['Outputs']    
    node_data = outputs['Output1']['nodeData']
    halo_masses = node_data['basicMass'][:]
    tree_weights = node_data['mergerTreeWeight'][:]
    z = node_data['redshift'][0]
    sfilename = f'spheroidLuminositiesStellar:{jwst_filter_name}:observed:z{z:.4f}'
    dfilename = f'diskLuminositiesStellar:{jwst_filter_name}:observed:z{z:.4f}'
    
    luminosity = node_data[sfilename][:] + node_data[dfilename][:] 
    # sets a lower limit on the luminosity to prevent 0 in log
    luminosity[luminosity<1.0e-10] = 1.0e-10 
    absolute_mag = -2.5 * np.log10(luminosity)
    apparent_mag = absolute_mag + cosmo.distmod(z).value - 2.5*np.log10(1+z)
    sorted_indices = np.argsort(halo_masses)
    halo_masses = halo_masses[sorted_indices]
    apparent_mag = apparent_mag[sorted_indices]
    absolute_mag = absolute_mag[sorted_indices]
    tree_weights = tree_weights[sorted_indices]
    redshifts = np.tile(z, len(halo_masses)) 
    data = np.stack((np.log10(halo_masses), apparent_mag, absolute_mag, redshifts, tree_weights),axis=-1)
    outfile.close()
    return data


def get_ngdeep_completeness(apparent_grid: NDArray) -> NDArray:
    """ Estimates the completeness function for the NGDEEP survey.
    
    Uses data from Table 2 of https://arxiv.org/pdf/2306.06244. Uses z~9 data,
    normalizes by the largest volume, and linearly interpolates between mag bins.
    Sets completeness to 0 at 5sigma limiting depth from Table 1.
    
    Args:
        apparent_grid: Array of apparent magnitude to evaluate completeness
          function on to.
    Returns:
        Array with the completeness function interpolated onto input apparent 
        magnitude grid.
    """
    survey_absolute_magnitudes = np.array([-21.1, -20.1, -19.1, -18.35, -17.85, -17.35])
    survey_apparent_magnitudes = (survey_absolute_magnitudes + 
                                  cosmo.distmod(9.0).value - 2.5*np.log10(1+9.0))
    effective_volumes = np.array([18700., 18500., 15800., 13100., 7770., 2520.]) # Mpc^3
  
    completeness = effective_volumes / np.amax(effective_volumes) 
    # make sure completeness goes to 0 at 5 sigma depth 
    apparent_magnitudes = np.append(survey_apparent_magnitudes, 30.4)
    completeness = np.append(completeness, 0.0)    
    # extends values to min and max of the grid for the purpose of interpolation 
    completeness = np.concatenate(([1.0], completeness, [0.0]))
    grid_min = np.amin(apparent_grid)
    grid_max = np.amax(apparent_grid)
    apparent_magnitudes = np.concatenate(([grid_min], apparent_magnitudes, 
                                          [grid_max]))
    completeness = np.interp(apparent_grid, apparent_magnitudes, completeness)
    return completeness


def get_ceers_completeness(apparent_grid: NDArray) -> NDArray:
    """ Estimates the completeness function for the CEERS survey.
    
    Uses data from Table 4 of https://arxiv.org/pdf/2311.04279. Uses z~9 data,
    normalizes by the largest volume, and linearly interpolates between mag bins.
    Sets completeness to 0 at 5sigma limiting depth from Table 1.
    
    Args:
        apparent_grid: Array of apparent magnitude to evaluate completeness
          function on to.
    Returns:
        Array with the completeness function interpolated onto input apparent 
        magnitude grid.
    """
    survey_absolute_magnitudes = np.array([-22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5])
    survey_apparent_magnitudes = (survey_absolute_magnitudes + 
                                  cosmo.distmod(9.0).value - 2.5*np.log10(1+9.0))
    effective_volumes = np.array([187000., 187000., 187000., 193000., 177000., 
                                  161000., 120000., 77900., 18600.]) # Mpc^3
    completeness = effective_volumes / np.amax(effective_volumes) 
    # make sure completeness goes to 0 at 5 sigma depth 
    apparent_magnitudes = np.append(survey_apparent_magnitudes, 29.15)
    completeness = np.append(completeness, 0.0)
    # extends values to min and max of the grid for the purpose of interpolation 
    completeness = np.concatenate(([1.0], completeness, [0.0]))
    grid_min = np.amin(apparent_grid)
    grid_max = np.amax(apparent_grid)
    apparent_magnitudes = np.concatenate(([grid_min], apparent_magnitudes, 
                                          [grid_max]))    
    cf = np.interp(apparent_magnitude_grid, apparent_magnitudes, completeness)
    return cf


def get_data_pdf(
    observed_data: pd.DataFrame, 
    apparent_grid: NDArray, 
    redshift_grid: NDArray, 
    filename: str, 
    load: bool = True,
    save: bool = True
) -> NDArray:
    """ Computes the pdf for galaxy candidates from redshift uncertainties.
    
    A probability density function for each galaxy candidate is computed assuming 
    a two-sided normal distribution using the medians and one sigma uncertainties 
    listed in https://arxiv.org/pdf/2306.06244 and https://arxiv.org/pdf/2311.04279. 
    A lower limit on the redshift of the pdf is placed at z=8.5 to mimic the 
    color cuts used in the selection of the high-z galaxy candidates after which
    the pdfs are renormalized.

    The pdfs are saved to file filename if save is True and will be loaded from the 
    same file if load is True.

    Args:
        observed_data: Pandas DataFrame with columns containing the apparent 
          magnitudes (mf277w), median redshifts (z), one sigma error above the 
          median (z_upper_err), and one sigma error below the mean (z_lower_err).
        apparent_grid: Grid of apparent magntiude on which the pdf is evaluated.
        redshift_grid: Grid of redshifts on which the pdf is evaluated.
        filename: Filename to save the the computed pdfs.
        load: if True, loads the data from filename.
        save: if True, saves the data to filename.
    Returns:
        Numpy array of the pdfs computed for each galaxy candidate, with shape 
        (n_gal, n_mag, n_z) where n_gal is the number galaxies in observed_data,
        n_mags is the length of the apparent_grid, and n_z is the length of 
        redshift_grid.
    """
    z_cutoff = 8.5 
    if load and path.isfile(filename):
        obs_pdf = np.load(filename)
    else:
        n_gal = len(observed_data)
        n_mag = len(apparent_grid)
        n_z = len(redshift_grid)
        obs_pdf = np.zeros((n_gal,n_mag,n_z))
        for i in range(n_gal):
            mag = observed_data['mf277w'][i]
            z = observed_data['z'][i]
            z_upper_err = observed_data['z_upper_err'][i]
            z_lower_err = np.abs(observed_data['z_lower_err'][i])
            dz = redshift_grid[1]-redshift_grid[0]
            
            mag_pdf = np.zeros_like(apparent_grid)
            idx = np.argmin(np.abs(apparent_grid-mag))
            mag_pdf[idx] = 1.0
            mag_pdf = mag_pdf.reshape(-1, 1)
            
            lower_idx = redshift_grid <= z # note, choice of <= is arbitrary!
            upper_idx = redshift_grid > z
            z_pdf = np.zeros_like(redshift_grid)
            norm = np.sqrt(2.0/np.pi)/(z_upper_err + z_lower_err)
            z_pdf[lower_idx] = norm * np.exp(-(redshift_grid[lower_idx]-z)**2 / 2.0 / z_lower_err**2)
            z_pdf[upper_idx] = norm * np.exp(-(redshift_grid[upper_idx]-z)**2 / 2.0 / z_upper_err**2)
            z_pdf = z_pdf.reshape(1, -1)
            
            obs_pdf[i,:,:] = mag_pdf * z_pdf * dz 
 
            cut_idx = redshift_grid < 8.5 # cutoff to mimic color cuts
            obs_pdf[i][:,cut_idx] = 0
            new_norm = np.sum(obs_pdf[i,:,:])
            obs_pdf[i,:,:] = obs_pdf[i,:,:] / new_norm # renormalized so sums to 1
        if save:
            np.save(filename, obs_pdf)
    return obs_pdf

    
def load_data(
    data_directory: str, 
    reload: bool = True, 
    save: bool = True,
) -> pd.DataFrame:
    """ Returns the data for one parameter combination. 
    
    Loads the 3 hdf5 files corresponding to different redshifts into arrays
    of log halo mass, absolute magnitudes, apparent magnitudes, and redshifts.

    Args:
        data_directory: Path to the directory containing the hdf5 files for the
          3 redshifts 
        reload: if True, force reloads the data 
        save: if True, saves the data to data_directory/data.csv

    Returns:
        Dataframe of galaxies where the columns contain the log halo mass, absolute
        magnitude, apparent magnitude, and redshift of all the simulated galaxies.
    """ 
    data_columns= ['log_halo_mass', 'apparent_magnitude', 'absolute_magnitude', 'redshift', 'tree_weight']
    data_filename = path.join(data_directory, 'data.csv')
    if not reload and path.isfile(data_filename):
        data = pd.read_csv(data_filename)
    else:
        data_list = []
        for z in ['8.0', '12.0', '16.0']: 
            hdf5_filename = path.join(data_directory,f'z{z}.hdf5')
            data = get_data_from_hdf5(hdf5_filename, 'JWST_NIRCAM_f277w')
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)
        data = pd.DataFrame(data, columns=data_columns)
        if save:
            data.to_csv(data_filename, index=False)
    return data


def get_binned_weights(
    logmh_bins: NDArray, 
    logmhs: NDArray, 
    weights: NDArray, 
    load: bool = True,
    save: bool = True,
) -> NDArray:
    """ Calculate binned merger tree weights.

    Args: 
        logmh_bins: Array of bin edges for the log halo mass bins.
        logmhs: Array of log halo masses for each galaxy.
        weights: Array of merger tree weights for each galaxy.
        load: if True, loads the data from data/binned_zgrid_weights.npy
        save: if True, saves the data to data/binned_zgrid_weights.npy
        
    Returns:
        Array of binned merger tree weights.
    """
    output_filename =  'data/binned_zgrid_weights.npy'
    if load and path.isfile(output_filename):
        binned_weights = np.load(output_filename)
    else:
        logmhs = np.unique(logmhs)
        binned_weights = np.zeros((nz,n_mass_bins))
        for i in range(n_mass_bins):
            left_bin_edge = Mh_bins[i]
            right_bin_edge = Mh_bins[i+1]
            bin_index = (logmhs > left_bin_edge) & (logmhs < right_bin_edge)
            binned_weights[:,i] = np.sum(weights[:,bin_index],axis=1)
        if save:
            np.save(output_filename, binned_weights)
    return binned_weights


def get_probs(
    magnitude_grid: NDArray, 
    stats: Stats, 
    data_directory: str, 
    do_abs: bool, 
    recompute: bool,
    save: bool=True,
) -> NDArray:
    """Returns an array of probabilities for the galaxy-halo connection.

    The pdf is determined by a normal distribution with mean and standard
    deviation determined from the simulated data for each halo mass bin and 
    redshift. The pdf is cutoff at a fraction of the min and max magnitudes to
    prevent artificial upscatter of faint galaxies. The pdf is saved to file if
    save is True. The output has shape (N_mag, N_z, N_mh).

    Args:
        mag_grid (ndarray): Magnitude grid.
        stats (namedtuple): Stats calculated from the data.
        data_directory (str): Directory to save data.
        do_abs (bool): Flag to determine absolute magnitude.
        recompute (bool): Flag to force recompute probabilities.
        save (bool, optional): Flag to save probabilities. Defaults to True.
    
    Returns:
        ndarray: Array of probabilities.
    """
    if do_abs:
        if stats is not None:
            mins = stats.absolute_min
            maxs = stats.absolute_max
            mean = stats.absolute_mean
            std = stats.absolute_sigma
            probability_min_magnitude = mins * 1.1 # absolute mags are negative
            probability_max_magnitude = maxs * 0.9
        probs_filename = path.join(data_directory, 'absolute_pdf.npy')
        # we need to cutoff pdf towards the faint end as distribution becomes 
        # non-gaussian to prevent artificial upscatter of faint galaxies 
        # (ideally would be fixed by using some skewed distribution)
        # values chosen based on empirical testing 
        
    else:
        if stats is not None:
            mins = stats.apparent_min
            maxs = stats.apparent_max
            mean = stats.apparent_mean
            std = stats.apparent_sigma
            probability_min_magnitude = mins * 0.95
            probability_max_magnitude = maxs * 1.05 
        probs_filename = path.join(data_directory, 'apparent_pdf.npy')
        # we need to cutoff pdf towards the faint end as distribution becomes 
        # non-gaussian to prevent artificial upscatter of faint galaxies 
        # (ideally would be fixed by using some skewed distribution)
        # values chosen based on empirical testing 
        
    if path.isfile(probs_filename) and not recompute:
        pdfs = np.load(probs_filename)
    else:
        magnitude_grid = magnitude_grid.reshape(-1,1,1)
        mean = mean.reshape(1, nz, n_mass_bins)
        std = std.reshape(1, nz, n_mass_bins)
        pdfs = np.exp(-(magnitude_grid-mean)**2/2.0/std**2)/np.sqrt(2*np.pi)/std
        
        for i, mag in enumerate(magnitude_grid):
            idx = (mag > probability_max_magnitude) | (mag < probability_min_magnitude)
            pdfs[i][idx] = 0

        # have to renormalize after cutting off the pdf
        norm = np.sum(pdfs, axis=0) 
        norm = norm.reshape(1, nz, n_mass_bins)        
        pdfs = (pdfs / norm)

        if save:
            np.save(probs_filename, pdfs)
    return pdfs


def get_skewed_probs(
    magnitude_grid: NDArray, 
    stats: Stats, 
    data_directory: str, 
    do_abs: bool, 
    recompute: bool,
    save: bool=True,
) -> NDArray:
    """Returns an array of probabilities for the galaxy-halo connection.

    The pdf is determined by a normal distribution with mean and standard
    deviation determined from the simulated data for each halo mass bin and 
    redshift. The pdf is cutoff at a fraction of the min and max magnitudes to
    prevent artificial upscatter of faint galaxies. The pdf is saved to file if
    save is True. The output has shape (N_mag, N_z, N_mh).

    Args:
        mag_grid (ndarray): Magnitude grid.
        stats (namedtuple): Stats calculated from the data.
        data_directory (str): Directory to save data.
        do_abs (bool): Flag to determine absolute magnitude.
        recompute (bool): Flag to force recompute probabilities.
        save (bool, optional): Flag to save probabilities. Defaults to True.
    
    Returns:
        ndarray: Array of probabilities.
    """
    if do_abs:
        # print(stats)
        if stats is not None: # temp 
            mu = stats.absolute_median
            sigma_L = stats.absolute_sigma_left
            sigma_R = stats.absolute_sigma_right
            probability_min_magnitude = stats.absolute_min * 1.1
            probability_max_magnitude = stats.absolute_max * 0.9
            # grid = absolute_magnitude_grid
            # nmuv = nabs
        probs_filename = path.join(data_directory, 'skewed_absolute_pdf.npy')
    else:
        if stats is not None: # temp 
            mu = stats.apparent_median
            sigma_L = stats.apparent_sigma_left
            sigma_R = stats.apparent_sigma_right
            # grid = apparent_magnitude_grid
            probability_min_magnitude = stats.apparent_min * 0.95
            probability_max_magnitude = stats.apparent_max * 1.05 
            # nmuv = napp
        probs_filename = path.join(data_directory, 'skewed_apparent_pdf.npy')
    if path.isfile(probs_filename) and not recompute:
        pdfs = np.load(probs_filename)
    else:
        pdfs = np.zeros((len(magnitude_grid), nz, n_mass_bins))
        for i in range(nz):
            for j in range(n_mass_bins):
                pdfs[:,i,j] = two_sided_normal_pdf(magnitude_grid, mu[i,j], sigma_L[i,j], sigma_R[i,j])
        # if do_abs:
        for i, mag in enumerate(magnitude_grid):
            idx = (mag > probability_max_magnitude) | (mag < probability_min_magnitude)
            pdfs[i][idx] = 0
        # could be built into two_sided_normal_pdf but just going to do it here as an explicit check
        norm = np.sum(pdfs, axis=0) 
        norm = norm.reshape(1, nz, n_mass_bins)        
        pdfs = (pdfs / norm)

        if save:
            np.save(probs_filename, pdfs)
    return pdfs


def get_uvlf(
    probs: NDArray, 
    binned_weights: NDArray,
    data_directory: str, 
    do_abs: bool, 
    do_skewed: bool,
    recompute: bool,
) -> NDArray:
    """ Returns the UVLF from the g-h connection pdfs and binned weights.

    Args: 
        probs: Array of probabilities.
        binned_weights: Array of binned weights.
        data_directory: Directory to save data.
        do_abs: Flag to determine absolute magnitude.
        do_skewed: Flag to prepend skewed to output fn
        recompute: Flag to force recompute probabilities.
    Returns:
        Array of the UVLF.
    """  
    if do_skewed:
        prefix = 'skewed_'
    else:
        prefix = ''
    if do_abs:
        uvlf_filename = path.join(data_directory, prefix+'absolute_uvlf.npy')
    else:
        uvlf_filename = path.join(data_directory, prefix+'apparent_uvlf.npy')
    if path.isfile(uvlf_filename) and not recompute:
        uvlf = np.load(uvlf_filename)
    else:
        uvlf = np.sum(binned_weights * probs, axis=2)
        np.save(uvlf_filename, uvlf)
    return uvlf


# def interpolate_uvlf(uvlf, time): 
#     # should I do this on log phi or phi and on time or z
    
#     pass


# def get_binned_uvlf(muvs, weights):
#     bin_width = 0.5
#     bins = np.arange(-24, -14, bin_width)
#     bin_centers = bins[:-1]+bin_width/2.0
#     left = bins[:-1]
#     right = bins[1:]
#     uvlf = np.zeros_like(bin_centers)
#     for i in range(len(bins)-1):
#         idx = (muvs < right[i]) & (muvs >= left[i])
#         uvlf[i] = np.sum(weights[idx]) / bin_width
#     return bin_centers, uvlf


def calculate_likelihood(apparent_uvlf: NDArray) -> float:
    """ Calculates the likelihood function given the UVLF.
    
    See https://arxiv.org/abs/2410.11680 for details. Note that input must be
    the UVLF in terms of apparent magnitude, not absolute.
    
    Args:
        apparent_uvlf: UVLF as a function of apparent magnitude and redshift
    Returns:
        log_likelihood (float): log of the likelihood (float)
    """
    z_cutoff = 8.5
    zidx = redshift_grid >= z_cutoff
    ngdeep_n = np.sum(apparent_uvlf[:,zidx]*ngdeep_effective_volume[:,zidx], dtype=np.longdouble) 
    ceers_n = np.sum(apparent_uvlf[:,zidx]*ceers_effective_volume[:,zidx], dtype=np.longdouble) 
    log_likelihood = -ngdeep_n-ceers_n
    ngdeep_obs_n = np.sum(ngdeep_pdf[:,:,zidx]*apparent_uvlf[:,zidx]*ngdeep_effective_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ngdeep_obs_n = np.log(ngdeep_obs_n[ngdeep_obs_n>0])
    ngdeep_obs_n = np.sum(ngdeep_obs_n)
    log_likelihood += ngdeep_obs_n
    ceers_obs_n = np.sum(ceers_pdf[:,:,zidx]*apparent_uvlf[:,zidx]*ceers_effective_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ceers_obs_n = np.log(ceers_obs_n[ceers_obs_n>0])
    ceers_obs_n = np.sum(ceers_obs_n)
    log_likelihood += ceers_obs_n
    return log_likelihood


# these are globals to save a little bit of time when running the calculations
# should probably get shoved into another file...
ngdeep_data = pd.read_csv('data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
zgrid_weights = get_weights_from_hmf('data/zgrid_hmfs.hdf5')

dMh = 0.15
Mh_bins = np.arange(8.0, 11.76, dMh)
global n_mass_bins
n_mass_bins = len(Mh_bins)-1
bin_centers = Mh_bins[:-1] + dMh/2.0 

global nz
nz = 17
global redshift_grid
redshift_grid = np.linspace(8.0, 16.0, nz)
dz = redshift_grid[1]-redshift_grid[0]

cv_z = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]
cv = [0.1522, 0.17411, 0.18843, 0.22613, 0.2549, 0.26993, 0.292834, 0.3769, 0.426021, 0.43414, 0.49211, 0.59527, 0.664436, 0.62814, 0.602, 0.602]
cv = np.interp(redshift_grid, cv_z, cv)

logmhs = np.log10(np.geomspace(1.0e8,5.0e11,3699))
global binned_weights
binned_weights = get_binned_weights(Mh_bins,logmhs,zgrid_weights, True)

abs_min = -25.0
abs_max = 0.
dabs = 0.2
nabs = int(round((abs_max-abs_min)/dabs))+1
absolute_magnitude_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = absolute_magnitude_grid[1]-absolute_magnitude_grid[0]

app_min = 22.0
app_max = 45.0
dapp = 0.25 
napp = int(round((app_max-app_min)/dapp))+1
apparent_magnitude_grid = np.linspace(app_min, app_max, napp) 
dapp = apparent_magnitude_grid[1]-apparent_magnitude_grid[0]

# global t
t_grid = cosmo.age(redshift_grid).value
t8 = cosmo.age(8.0).value
t12 = cosmo.age(12.0).value
t16 = cosmo.age(16.0).value

z_volumes = (cosmo.comoving_volume(redshift_grid+dz/2.0)-cosmo.comoving_volume(redshift_grid-dz/2.0)).value 

ngdeep_cf = get_ngdeep_completeness(apparent_magnitude_grid)
ngdeep_area = 3.3667617763401435e-08 # 5 arcmin^2 as a fraction of the sky
ngdeep_volumes = ngdeep_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
ngdeep_effective_volume = ngdeep_cf.reshape(-1,1)*ngdeep_volumes.reshape(1,-1)
ngdeep_effective_volume = np.abs(ngdeep_effective_volume)

ceers_cf = get_ceers_completeness(apparent_magnitude_grid)
ceers_area = 5.932234249911333e-07 # 88.1 arcmin^2 as a fraction of the sky
ceers_volumes = ceers_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
ceers_effective_volume = ceers_cf.reshape(-1,1)*ceers_volumes.reshape(1,-1)
ceers_effective_volume = np.abs(ceers_effective_volume)

ngdeep_pdf_filename =  'data/ngdeep_pdf.npy'
ceers_pdf_filename = 'data/ceers_pdf.npy'
ngdeep_pdf = np.abs(get_data_pdf(ngdeep_data, apparent_magnitude_grid, redshift_grid, ngdeep_pdf_filename, True))
ceers_pdf = np.abs(get_data_pdf(ceers_data, apparent_magnitude_grid, redshift_grid, ceers_pdf_filename, True))


def two_sided_normal_pdf(x, mu, sigma_L, sigma_R):
    return np.where(x < mu,
                    scipy.stats.norm.pdf(x, loc=mu, scale=sigma_L),
                    scipy.stats.norm.pdf(x, loc=mu, scale=sigma_R))


# Define the negative log-likelihood function
def neg_log_likelihood(params, x):
    mu, sigma_L, sigma_R = params
    sigma_L, sigma_R = abs(sigma_L), abs(sigma_R)  # Ensure positive values
    
    likelihoods = two_sided_normal_pdf(x, mu, sigma_L, sigma_R)
    return -np.sum(np.log(likelihoods + 1e-10)) 


def get_skewed_stats(data: pd.DataFrame) -> Stats:
    sampled_redshifts = np.unique(data['redshift'])[::-1]
    sampled_nz = len(sampled_redshifts)
    sampled_absolute_medians = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_sigma_lefts = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_sigma_rights = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_mins = np.zeros((sampled_nz, n_mass_bins))

    sampled_apparent_medians = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_sigma_lefts = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_sigma_rights = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_mins = np.zeros((sampled_nz, n_mass_bins))
    for i,z in enumerate(sampled_redshifts):
        zidx = data['redshift']==z
        z_lmh = data['log_halo_mass'][zidx]
        z_abs = data['absolute_magnitude'][zidx]
        z_app = data['apparent_magnitude'][zidx]

        for j in range(n_mass_bins):
            left = Mh_bins[j]
            right = Mh_bins[j+1]
            bidx = (z_lmh > left) & (z_lmh < right)

            abs_samples = z_abs[bidx]
            mu_init = np.median(abs_samples)
            sigma_L_init = np.std(abs_samples[abs_samples < mu_init])
            sigma_R_init = np.std(abs_samples[abs_samples >= mu_init])
            abs_result = scipy.optimize.minimize(
                neg_log_likelihood, 
                x0=[mu_init, sigma_L_init, sigma_R_init], 
                args=(abs_samples,),
                method='L-BFGS-B', 
                bounds=[(-30, 0), (1e-2, 1e2), (1e-2, 1e2)]
                )
            sampled_absolute_medians[i,j] = abs_result.x[0]
            sampled_absolute_sigma_lefts[i,j] = abs_result.x[1]
            sampled_absolute_sigma_rights[i,j] = abs_result.x[2]
            sampled_absolute_maxs[i,j] = np.amax(abs_samples)
            sampled_absolute_mins[i,j] = np.amin(abs_samples)

            app_samples = z_app[bidx]
            mu_init = np.median(app_samples)
            sigma_L_init = np.std(app_samples[app_samples < mu_init])
            sigma_R_init = np.std(app_samples[app_samples >= mu_init])
            app_result = scipy.optimize.minimize(
                neg_log_likelihood, 
                x0=[mu_init, sigma_L_init, sigma_R_init], 
                args=(app_samples,),
                method='L-BFGS-B', 
                bounds=[(0, 50), (1e-2, 1e2), (1e-2, 1e2)]
                )
            sampled_apparent_medians[i,j] = app_result.x[0]
            sampled_apparent_sigma_lefts[i,j] = app_result.x[1]
            sampled_apparent_sigma_rights[i,j] = app_result.x[2] 
            sampled_apparent_maxs[i,j] = np.amax(app_samples)
            sampled_apparent_mins[i,j] = np.amin(app_samples)        

    absolute_medians = np.zeros((nz, n_mass_bins))
    absolute_sigma_lefts = np.zeros((nz, n_mass_bins))
    absolute_sigma_rights = np.zeros((nz, n_mass_bins))
    absolute_maxs = np.zeros((nz, n_mass_bins))
    absolute_mins = np.zeros((nz, n_mass_bins))

    apparent_medians = np.zeros((nz, n_mass_bins))
    apparent_sigma_lefts = np.zeros((nz, n_mass_bins))
    apparent_sigma_rights = np.zeros((nz, n_mass_bins))
    apparent_maxs = np.zeros((nz, n_mass_bins))
    apparent_mins = np.zeros((nz, n_mass_bins))

    sampled_times = [t16, t12, t8]
    for j in range(n_mass_bins):
        absolute_medians[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_medians[:,j])
        absolute_sigma_lefts[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_sigma_lefts[:,j])
        absolute_sigma_rights[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_sigma_rights[:,j])
        absolute_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_maxs[:,j])
        absolute_mins[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_mins[:,j])

        apparent_medians[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_medians[:,j])
        apparent_sigma_lefts[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_sigma_lefts[:,j])
        apparent_sigma_rights[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_sigma_rights[:,j])
        apparent_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_maxs[:,j])
        apparent_mins[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_mins[:,j])

    stats = Stats(absolute_medians, absolute_sigma_lefts, absolute_sigma_rights, absolute_mins, absolute_maxs,
                    apparent_medians, apparent_sigma_lefts, apparent_sigma_rights, apparent_mins, apparent_maxs)
    return stats


def get_stats(data: pd.DataFrame) -> Stats:
    """ Returns dataframe containing stats calculated from the data.

    """
    sampled_redshifts = np.unique(data['redshift'])[::-1]
    sampled_nz = len(sampled_redshifts)
    sampled_absolute_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_mins = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_means = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_stds = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_mins = np.zeros((sampled_nz, n_mass_bins))
    for i,z in enumerate(sampled_redshifts):
        zidx = data['redshift']==z
        z_lmh = data['log_halo_mass'][zidx]
        z_abs = data['absolute_magnitude'][zidx]
        z_app = data['apparent_magnitude'][zidx]

        for j in range(n_mass_bins):
            left = Mh_bins[j]
            right = Mh_bins[j+1]
            bidx = (z_lmh > left) & (z_lmh < right)
            # if np.sum(bidx)==0:
            #     sampled_absolute_means[i,j] = -3
            #     sampled_absolute_stds[i,j] = 1
            #     sampled_absolute_maxs[i,j] = -1
            #     sampled_absolute_mins[i,j] = -5
            #     sampled_apparent_maxs[i,j] = 45
            #     sampled_apparent_mins[i,j] = 35
            # else:
            sampled_absolute_means[i,j] = np.average(z_abs[bidx])
            sampled_absolute_stds[i,j] = np.std(z_abs[bidx])
            sampled_absolute_maxs[i,j] = np.amax(z_abs[bidx])
            sampled_absolute_mins[i,j] = np.amin(z_abs[bidx])
            sampled_apparent_maxs[i,j] = np.amax(z_app[bidx])
            sampled_apparent_mins[i,j] = np.amin(z_app[bidx])

    absolute_sigmas = np.zeros((nz, n_mass_bins))
    absolute_means = np.zeros((nz, n_mass_bins))
    absolute_maxs = np.zeros((nz, n_mass_bins))
    absolute_mins = np.zeros((nz, n_mass_bins))
    apparent_maxs = np.zeros((nz, n_mass_bins))
    apparent_mins = np.zeros((nz, n_mass_bins))

    sampled_times = [t16, t12, t8]
    for j in range(n_mass_bins):
        absolute_means[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_means[:,j])
        absolute_sigmas[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_stds[:,j])
        absolute_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_maxs[:,j])
        absolute_mins[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_mins[:,j])
        apparent_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_maxs[:,j])
        apparent_mins[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_mins[:,j])
    shift = cosmo.distmod(redshift_grid).value-2.5*np.log10(1.+redshift_grid)
    apparent_means = absolute_means + shift.reshape(-1,1)
    apparent_sigmas = absolute_sigmas.copy()
    stats = Stats(absolute_means, absolute_sigmas, absolute_mins, absolute_maxs,
                    apparent_means, apparent_sigmas, apparent_mins, apparent_maxs)
    return stats


def evaluate_likelihood(i, data_directory, skewed, reload, recompute):
    """ Returns the log likelihood for a given parameter combination. 
    
    If any error is thrown during the calculation (e.g. data is not available), 
    the function will return np.nan and the error will be prined to the console."""
    # try:
    data = load_data(data_directory, reload)
    uvlf_filename = path.join(data_directory, 'apparent_uvlf.npy')
    if not recompute and path.isfile(uvlf_filename):
        abs_probs = None 
        app_probs = None
    else:
        if skewed:
            stats = get_skewed_stats(data)        
            abs_probs = get_skewed_probs(absolute_magnitude_grid, stats, data_directory, True, recompute)
            app_probs = get_skewed_probs(apparent_magnitude_grid, stats, data_directory, False, recompute)
        else:
            stats = get_stats(data)        
            abs_probs = get_probs(absolute_magnitude_grid, stats, data_directory, True, recompute)
            app_probs = get_probs(apparent_magnitude_grid, stats, data_directory, False, recompute)

    # the absolute uvlf is not used in the calculation but is computed for plotting purposes so it is computed here
    abs_uvlf = get_uvlf(abs_probs, binned_weights, data_directory, True, skewed, recompute)/dabs
    app_uvlf = get_uvlf(app_probs, binned_weights, data_directory, False, skewed, recompute)
    loglike = calculate_likelihood(app_uvlf)
    # except Exception as e:
    #     print(i)
    #     print(e)
    #     loglike = np.nan
    return loglike


def get_astro_params(dirname, initial, final):
    base = '/carnegie/nobackup/users/gdriskell/jwst_data/'
    Vout_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
    alphaOut_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
    tau0_xml = 'starFormationRateDisks/starFormationTimescale/timescale'
    alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/exponentVelocity'

    outflow_velocities = []
    outflow_alphas = []
    sfr_alphas = []
    sfr_timescales = []

    for i in range(initial, final):
        data_directory = path.join(base, dirname+f'_p{i}/')
        xml_filename = path.join(data_directory,'z8.0.xml')
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        outflow_velocities.append(float(root.find(Vout_xml).get('value')))
        outflow_alphas.append(float(root.find(alphaOut_xml).get('value')))
        sfr_timescales.append(float(root.find(tau0_xml).get('value')))
        sfr_alphas.append(float(root.find(alphaStar_xml).get('value')))
        
    return outflow_velocities, outflow_alphas, sfr_alphas, sfr_timescales


def run(dirname, base, i, skewed, reload, recompute):
    data_directory = path.join(base, dirname+f'_p{i}/')
    loglike = evaluate_likelihood(i, data_directory, skewed, reload, recompute)
    return loglike


def save_results(loglikes, initial, final, dirname, outfilename):
    (outflow_velocities, outflow_alphas, sfr_alphas, 
            sfr_timescales) = get_astro_params(dirname, initial, final+1)
    idxs = list(range(initial, final+1))
    columns = ['outflow_velocity', 'outflow_alpha', 'sfr_timescale', 
                'sfr_alpha', 'loglike']
    data = np.array([outflow_velocities, outflow_alphas, sfr_timescales, 
                        sfr_alphas, loglikes]).T
    df = pd.DataFrame(data, columns=columns, index=idxs)
    if len(outfilename)>0:
        output_csv_filename = f'{outfilename}.csv'
    else:
        output_csv_filename = f'{dirname}.csv'
    print(f'Saving to {output_csv_filename}')

    df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
    df.insert(len(df.columns), 'like', np.exp(df['loglike']))
    df = df.sort_values('loglike', ascending=False)

    df.to_csv(output_csv_filename)    
    print(df.head(n=10))


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("dirname", help="Path to data directory")
    parser.add_argument("--base", type=str,  help="Base directory for data files")
    parser.add_argument("--outfilename", type=str, default='', help="Output filename")
    parser.add_argument("--initial", type=int, help="Initial index to run")
    parser.add_argument("--final", type=int, help="Final index to run")
    parser.add_argument("--save", action='store_true', help="Whether to save output to file")
    parser.add_argument("--skewed", action='store_true', help="Whether to use skewed pdf for analysis")
    parser.add_argument("--reload", action='store_true', help="Whether to reload data")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute probs and uvlf")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    start = time()
    loglikes = Parallel(n_jobs=args.n_jobs)(delayed(run)(args.dirname, args.base, i, args.skewed,
         args.reload, args.recompute,) for i in range(args.initial, args.final+1))
    print(f'for n={args.final-args.initial+1} total time = {time()-start}')

    if args.save:
       save_results(loglikes, args.initial, args.final, args.dirname, 
                    args.outfilename)

