
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
import yaml 
# import corner
import pandas as pd
from scipy.cluster.vq import kmeans
import glob
from multiprocessing import Pool
import pickle 
import xml.etree.ElementTree as ET

import tensorflow as tf

num_threads = 16
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "16"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

import tensorflow_probability as tfp
import gpflow as gpf
from gpflow.ci_utils import reduce_in_tests
import gpr_util

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
        for i in range(21):
            j = 21-i
            output = f[f'Outputs/Output{j}']
            haloMass = output['haloMass'][:]
            hmf = output['haloMassFunctionM'] 
            weight = deltaM*hmf
            weights.append(weight)
        weights = np.array(weights)
        f.close()
        np.save(weights_fn,weights)
    return weights

global cosmo
cosmo = FlatLambdaCDM(H0=67.36000, Om0=0.31530, Tcmb0=2.72548, Ob0=0.04930)    
        

def get_heteroskedastic_model(Z, variance, lengthscales, do_abs):
    # kernel_type = gpf.kernels.RBF
    kernel_type = gpf.kernels.Matern12
    # likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    #     distribution_class=tfp.distributions.Gumbel,  
    #     scale_transform=tfp.bijectors.Softplus(low=tf.constant(1.0e-1,dtype=tf.float64)),
    # ) 
    likelihood = gpr_util.TwoPieceNormalLikelihood()    
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_type(variance=variance[0], lengthscales=lengthscales[0]),  
            kernel_type(variance=variance[1], lengthscales=lengthscales[1]),  
            kernel_type(variance=variance[2], lengthscales=lengthscales[2]),  
        ]
    )
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z), 
            gpf.inducing_variables.InducingPoints(Z),
            gpf.inducing_variables.InducingPoints(Z), 
        ]
    )
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
    )
    return model


def train_model(model, train_iter, niter, gamma, adam_lr, param_fn, data_size):
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=gamma)
    variational_params = [(model.q_mu, model.q_sqrt)]
    gpf.set_trainable(model.q_mu, False)
    gpf.set_trainable(model.q_sqrt, False)
    maxiter = reduce_in_tests(niter)
    training_loss = model.training_loss_closure(train_iter, compile=True)


    adam_opt = tf.optimizers.Adam(adam_lr)
    
    prev_loss = []
    best_loss = np.inf

    # @tf.function
    def optimization_step():
        natgrad_opt.minimize(training_loss, variational_params) # 
        adam_opt.minimize(training_loss, model.trainable_variables)

    for step in range(maxiter):
        optimization_step()
        loss = training_loss().numpy()/data_size
        prev_loss.append(loss)
            
        if step % 100 == 0:
            avg_loss = np.average(prev_loss)
            print(avg_loss)
            if avg_loss < best_loss:
                params = gpf.utilities.parameter_dict(model)
                f = open(param_fn,'wb')
                pickle.dump(params,f)
                f.close()
                best_loss = avg_loss
            prev_loss=[]

    gpf.utilities.print_summary(model)
    return model


def get_sparse_model(X_train, Y_train, data_dir, do_abs=False, task='load', output_tag=''):
    N, D = X_train.shape
    kernel_variance = np.array([0.05, 0.05, 0.05], dtype=np.float64)
    lengthscales = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],dtype=np.float64)
    n_inducing = 45 # 40
    inducing_variable, _ = kmeans(X_train, n_inducing)
    Y_train = Y_train.reshape(-1, 1)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(N))
    model = get_heteroskedastic_model(inducing_variable, kernel_variance, lengthscales, do_abs)
    if do_abs:
        model_dir = path.join(data_dir, 'abs_model_'+output_tag)
    else: 
        model_dir = path.join(data_dir, 'app_model_'+output_tag)
    if task=='load' or task=='continue': 
        with open(model_dir+"_params.pkl","rb") as f:
            params = pickle.load(f)
        gpf.utilities.multiple_assign(model, params)
    if task =='continue' or task=='retrain':
        param_fn = model_dir+'_params.pkl'
        for lr in [0.05, 0.01]:
            print(f'lr = {lr}')
            nit = 1000 # 1100
            model = train_model(model, train_iter, nit, lr/100.0, lr, param_fn, N)
    return model


# def plot_fit_and_data(ax, x_data, y_data, x_plot, output_mean, output_var, c='C0'):
#     ax.plot(x_data, y_data, "kx", mew=1, label="Sim. data")
#     ax.plot(x_plot, output_mean, "-", lw=3, label="Fit mean", c=c)
#     output_upper = output_mean + 1.96 * np.sqrt(output_var)
#     output_lower = output_mean - 1.96 * np.sqrt(output_var)
#     ax.plot(x_plot, output_lower, ".", label=r"Fit $95\%$ confidence", c=c)
#     ax.plot(x_plot, output_upper, ".", c=c)
#     ax.fill_between(
#         x_plot, output_lower[:, 0], output_upper[:, 0], color=c, alpha=0.3
#     )

def plot_fit_and_data_density(ax, x_data, y_data, x_plot, density, mag_grid, c='C0'):
    # density is muv, z, mh
    density = density[:,0,:]
    
    nmuv, nmh = density.shape
    ax.plot(x_data, y_data, "kx", mew=1, label="Sim. data")
    mode_density = np.amax(density, axis=0)
    
    density = density / mode_density #.reshape(1,-1)
    output_lower = np.zeros(nmh)
    output_upper = np.zeros(nmh)
    modes = np.zeros(nmh)
    for j in range(nmh):
        dj = density[:,j]
        idx = np.where(np.isclose(dj,1., rtol=1e-8))[0][0]
        modes[j] = mag_grid[idx]
        # print(idx)
        lower_density = dj[:idx]
        # print(lower_density)
        lower_twosigma = lower_density < np.exp(-2)
        if idx==0:
            output_lower[j] =  mag_grid[0]
        elif np.any(lower_twosigma):
            output_lower[j] = mag_grid[:idx][lower_twosigma][-1] # lower_density[lower_twosigma][-1]
        else:
            output_lower[j] = mag_grid[:idx][0] #lower_density[0]
        upper_density = dj[idx+1:]
        # print(upper_density)
        upper_twosigma = upper_density < np.exp(-2)
        # print(upper_twosigma)
        if idx == (len(mag_grid)-1):
            output_upper[j] = mag_grid[-1]   
        elif np.any(upper_twosigma):
            output_upper[j] = mag_grid[idx+1:][upper_twosigma][0]# upper_density[upper_twosigma][0]
        else:
            output_upper[j] = mag_grid[idx+1:][-1]
    ax.plot(x_plot, modes, "-", lw=3, label="Fit mode", c=c)
    ax.plot(x_plot, output_lower, ".", label=r"Approx. $95\%$ confidence", c=c)
    ax.plot(x_plot, output_upper, ".", c=c)
    ax.fill_between(
        x_plot, output_lower, output_upper, color=c, alpha=0.3
    )


def plot_fit_density(ax, x_plot, density, mag_grid, c='C0'):
    # density is muv, z, mh
    density = density[:,0,:]
    
    nmuv, nmh = density.shape
    # ax.plot(x_data, y_data, "kx", mew=1, label="Sim. data")
    mode_density = np.amax(density, axis=0)
    
    density = density / mode_density #.reshape(1,-1)
    output_lower = np.zeros(nmh)
    output_upper = np.zeros(nmh)
    modes = np.zeros(nmh)
    for j in range(nmh):
        dj = density[:,j]
        idx = np.where(np.isclose(dj,1., rtol=1e-8))[0][0]
        modes[j] = mag_grid[idx]
        # print(idx)
        lower_density = dj[:idx]
        # print(lower_density)
        lower_twosigma = lower_density < np.exp(-2)
        if idx==0:
            output_lower[j] =  mag_grid[0]
        elif np.any(lower_twosigma):
            output_lower[j] = mag_grid[:idx][lower_twosigma][-1] # lower_density[lower_twosigma][-1]
        else:
            output_lower[j] = mag_grid[:idx][0] #lower_density[0]
        upper_density = dj[idx+1:]
        # print(upper_density)
        upper_twosigma = upper_density < np.exp(-2)
        # print(upper_twosigma)
        if idx == (len(mag_grid)-1):
            output_upper[j] = mag_grid[-1]   
        elif np.any(upper_twosigma):
            output_upper[j] = mag_grid[idx+1:][upper_twosigma][0]# upper_density[upper_twosigma][0]
        else:
            output_upper[j] = mag_grid[idx+1:][-1]
    ax.plot(x_plot, modes, "-", lw=3, label="Fit mode", c=c)
    ax.plot(x_plot, output_lower, ".", label=r"Approx. $95\%$ confidence", c=c)
    ax.plot(x_plot, output_upper, ".", c=c)
    ax.fill_between(
        x_plot, output_lower, output_upper, color=c, alpha=0.25
    )


def plot_model_output(mh_cutoff, X, Y, model, data_dir, mag_grid, plot_abs=True, output_tag=''):
    zs = [8.0, 12.0, 16.0]
    f, axs = plt.subplots(len(zs), 1, figsize=(6,15),constrained_layout=True)
    m_max =  np.amax(X[:,0])
    plot_mhs = np.linspace(mh_cutoff,m_max, 50).reshape(50,1)
    for i,z in enumerate(zs):
        ax = axs[i]
        # Xplot = np.hstack((plot_mhs,np.ones_like(plot_mhs)*z))
        # y_mean, y_var = model.predict_y(Xplot) # model.compiled_predict_y(Xplot)
        density = get_stats_from_model(model, plot_mhs, [z], mag_grid) 
        # print(X[:,-1])
        idx = X[:,-1]==z
        # print(idx)
        plot_fit_and_data_density(ax, X[idx][:,0], Y[idx], plot_mhs.flatten(), density, mag_grid)
        ax.set_title(f'$z={z}$')
        ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
        
        ax.set_xlim(mh_cutoff,m_max)
        # ax.set_ylabel(r'$m_{\mathrm{UV}}$')
        if plot_abs:
            ax.set_ylim(-10,-25)
            ax.set_ylabel(r'$M_{\mathrm{UV}}$')
        else:
            ax.set_ylim(31.5, 22)
            ax.set_ylabel(r'$m_{\mathrm{UV}}$')
    if plot_abs:
        plt.savefig(path.join(data_dir, output_tag+'_abs_model_output.pdf'))
    else:
        plt.savefig(path.join(data_dir, output_tag+'_app_model_output.pdf'))
    plt.close('all')


def plot_interp_output(mh_cutoff, model, data_dir, mag_grid, plot_abs=True, output_tag=''):
    zs = [8.0, 10.0, 12.0, 14.0, 16.0]
    f, ax = plt.subplots(1, 1, figsize=(6,5),constrained_layout=True)
    plot_mhs = np.linspace(mh_cutoff, 11.5, 25).reshape(25,1)
    norm = mpl.colors.Normalize(vmin=np.amin(zs)-0.5, vmax=np.amax(zs)+0.5)
    sm = mpl.cm.ScalarMappable(norm, cmap='viridis')
    for i,z in enumerate(zs):
        # ax = axs[i]
        
        density = get_stats_from_model(model, plot_mhs, [z], mag_grid) 

        # color = 
        
        
        plot_fit_density(ax, plot_mhs.flatten(), density, mag_grid, c=sm.to_rgba(z))
        # ax.set_title(f'$z={z}$')
        ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
        
        ax.set_xlim(mh_cutoff,11.5)
        if plot_abs:
            ax.set_ylim(-10,-25)
            ax.set_ylabel(r'$M_{\mathrm{UV}}$')
        else:
            ax.set_ylim(31.5, 22)
            ax.set_ylabel(r'$m_{\mathrm{UV}}$')
    if plot_abs:
        plt.savefig(path.join(data_dir, output_tag+'_abs_interp_output.pdf'))
    else:
        plt.savefig(path.join(data_dir, output_tag+'_app_interp_output.pdf'))
    plt.close('all')



def plot_data_hist(mh_cutoff, X, Y, model, data_dir, mag_grid, plot_abs=True, output_tag=''):
    zs = [8.0, 12.0, 16.0]
    colors = [plt.cm.Dark2(i) for i in range(8)]
    f, axs = plt.subplots(len(zs), 1, figsize=(6,15),constrained_layout=True)
    mh_min = int(round(mh_cutoff))
    plot_mhs = np.arange(mh_min+1, 12)
    muv_bins = np.arange(min(mag_grid), max(mag_grid)+0.3, 0.25)
    hist_samples = 200
    plot_log = False
    for i,z in enumerate(zs):
        ax = axs[i]
        idx = X[:,-1]==z
        for j,m in enumerate(plot_mhs):
            m_idx = np.argmin(np.abs(X[idx][:,0]-m))
            # print(X[idx][m_idx][0], m)
            if m_idx < hist_samples:
                lidx = 0
                ridx = hist_samples
            elif (len(X[idx,:]) - m_idx) < hist_samples:
                lidx = -hist_samples
                ridx = len(X[idx])
            else:
                lidx = m_idx-hist_samples//2
                ridx = m_idx+hist_samples//2
            # print(len(plot_Y))
            plot_x = X[idx][lidx:ridx,0]
            plot_Y = Y[idx][lidx:ridx]
            density = get_stats_from_model(model, plot_x, [z], muv_bins)[:,0,:]
            # ax.hist(plot_Y, density=True, log=True, color=colors[j], alpha=0.35, zorder=-999, bins=muv_bins)
            density /= (np.sum(density,axis=0)*0.25)
            plot_density = np.mean(density,axis=1)
            if plot_log:
                # log_density -= np.log(np.sum(density,axis=0)*(muv_bins[1]-muv_bins[0]))
                ax.hist(plot_Y, density=True, log=True, color=colors[j], alpha=0.35, zorder=-999, bins=muv_bins)
                ax.semilogy(muv_bins, plot_density, color=colors[j], label=r'$\mathrm{Log}(M_{h})='+str(m)+'$', zorder=999)
                # print(np.log10(plot_density))

            else:#/hist_samples
                ax.hist(plot_Y, density=True, color=colors[j], alpha=0.35, zorder=-999, bins=muv_bins)
                ax.plot(muv_bins, plot_density, color=colors[j], label=r'$\mathrm{Log}(M_{h})='+str(m)+'$', zorder=999)

        ax.set_title(f'$z={z}$')
        ax.legend(frameon=False)
        # ax.set_ylabel(r'$m_{\mathrm{UV}}$')
        if plot_abs:
            # ax.set_xlim(-10,-25)
            ax.set_xlabel(r'$M_{\mathrm{UV}}$')
            ax.set_ylabel(r'$P(M_{\mathrm{UV}})$')
        else:
            # ax.set_xlim(35, 22)
            ax.set_xlabel(r'$m_{\mathrm{UV}}$')
            if plot_log: 
                ax.set_ylabel(r'$\mathrm{Log}P(m_{\mathrm{UV}})$')
                ax.set_ylim(1e-2, 2.0)
            else:
                ax.set_ylabel(r'$P(m_{\mathrm{UV}})$')     
    if plot_abs:
        plt.savefig(path.join(data_dir, output_tag+'_abs_hist.pdf'))
    else:
        if plot_log:
            plt.savefig(path.join(data_dir, output_tag+'_app_log_hist.pdf'))
        else:
            plt.savefig(path.join(data_dir, output_tag+'_app_hist.pdf'))
    plt.close('all')


def test_model_diff(X, Y, data_dir, mag_grid, weights, model, z_grid, output_tag=''):
    # focusing on data part for now
    zs = [8.0, 12.0, 16.0]
    colors = [plt.cm.Dark2(i) for i in range(8)]
    f, axs = plt.subplots(len(zs), 1, figsize=(6,15),constrained_layout=True)
    # mh_min = int(round(mh_cutoff))
    lmh_bins = np.linspace(8.5, 11, 35)
    logmhs = np.log10(np.logspace(7,11,4001)[:-1])+0.0005 #np.unique(X[:,0])
    new_weights = np.zeros((len(lmh_bins), len(zs)))
    # print(weights,sha)

    for i,lmh in enumerate(lmh_bins):
        idx = np.argmin(np.abs(logmhs-lmh))
        for j,z in enumerate(zs):
            zidx = np.argmin(np.abs(z_grid-z))
            new_weights[i,j] = weights[zidx,idx]
    plot_muvs = [27.0, 28.0, 29.0]

    for i,z in enumerate(zs):
        ax = axs[i]
        idx = X[:,-1]==z
        for j,m in enumerate(plot_muvs):
            left_muv = m - 0.25
            right_muv = m + 0.25
            ### model part
            density_muvs = np.linspace(left_muv,right_muv,10)
            probs = get_stats_from_model(model, lmh_bins, [z], density_muvs)
            probs = (np.sum(probs,axis=0)/10.0).flatten()
            # probs 
            # logmhs = np.unique(logmhs)
            mh_probs = new_weights[:,i] / np.sum(new_weights[:,i])
            # npad = new_weights.shape[-1]-probs.shape[-1]
            # probs = np.pad(probs, [[0,0],[0,0],[npad,0]])
            muv_prob = np.sum(mh_probs * probs) # muv x z
            prob_mh_given_muv = mh_probs * probs / muv_prob
            prob_mh_given_muv /= (np.sum(prob_mh_given_muv)*(lmh_bins[1]-lmh_bins[0]))
            ###

            Xz = X[idx]
            Yz = Y[idx]
            
            midxs = ((Yz > left_muv) & (Yz < right_muv))
            plot_Y = Yz[midxs]
            plot_X = Xz[midxs][:,0]
            plot_weights = np.zeros_like(plot_X)
            for k,x in enumerate(plot_X):
                midx = np.argmin(np.abs(logmhs - x))
                zidx = np.argmin(np.abs(z_grid-z))
                plot_weights[k] = weights[zidx,midx]
            # plot_weights = 
            # print(plot_X.shape)
            ax.hist(plot_X, weights=plot_weights, density=True, color=colors[j], alpha=0.35, zorder=-999, bins=lmh_bins)
            ax.plot(lmh_bins, prob_mh_given_muv, color=colors[j], label=r'$m_{\mathrm{uv}}='+str(m)+'$', zorder=999)
        ax.set_title(f'$z={z}$')
        ax.legend(frameon=False)
        ax.set_xlabel(r'$\mathrm{Log}(M_{h})$')
        ax.set_ylabel(r'$P(\mathrm{Log}(M_{h})|M_{\mathrm{uv}})$')
    plt.savefig(path.join(data_dir, output_tag+'_app_mh_hist.pdf'))
    plt.close('all')
  

# def plot_model_and_weights(mh_cutoff, X, Y, model, data_dir, weights, plot_abs=True):
#     # zs = [8.0]
#     colors = [plt.cm.Dark2(i) for i in range(8)]
#     f, ax = plt.subplots(figsize=(7.0,4.8))
#     lmhs = np.log10(np.logspace(7,11,4001)[:-1])+0.0005
#     widx = lmhs>=9
#     lmhs = lmhs[widx]
#     plot_mhs = np.linspace(mh_cutoff, 11, 50).reshape(50,1)
#     Xplot = np.hstack((plot_mhs,np.ones_like(plot_mhs)*8.0))
#     y_mean, y_var = model.predict_y(Xplot) # model.compiled_predict_y(Xplot)
#     idx = X[:,-1]==8.0
#     c1 = colors[0]
#     c2 = colors[2]
#     plot_fit_and_data(ax, X[idx,0], Y[idx], Xplot[:,0], y_mean, y_var, c=c1)
#     ax2 = ax.twinx()
#     ax2.semilogy(lmhs, weights[5,widx], lw=3, color=c2)
#     ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
    
#     ax.set_xlim(9.0,11)
#     if plot_abs:
#         ax.set_ylim(-10,-25)
#         ax.set_ylabel(r'$M_{\mathrm{UV}}$', color=c1)
        
#     else:
#         ax.set_ylim(35, 22)
#         ax.set_ylabel(r'$m_{\mathrm{UV}}$')
#     ax2.set_ylabel(r'$n(M_{h})$', color=c2)
#     ax2.minorticks_off()

#     ax.legend(loc='lower left', frameon=False, fontsize=12.5)
#     plt.tight_layout()
#     # plt.savefig('data_and_fit.png',dpi=500)
#     plt.savefig('model_output_and_weights.png',dpi=500)

#     plt.close('all')

def get_stats_from_model(model, logmhs, z_grid, mag_grid):
    """ Output shape is n_muv, n_z, n_mh """
    logmhs = np.unique(logmhs)
    nx = len(logmhs)
    ny = len(z_grid)
    nz = len(mag_grid)
    # print(nx,ny,nz)

    X, Y, Z = np.meshgrid(logmhs, z_grid, mag_grid, indexing='ij') # out shape with be nx, ny, nz
    input_data = np.hstack([X.reshape(-1,1), Y.reshape(-1,1)]) 
    # print(input_data.shape)
    # print(X.shape, Y.shape, Z.shape)
    # data = [input_data, Z.reshape(-1,1)] # from https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points
    # muv_mean, muv_var = model.predict_y(data)
    # print(data.shape)
    # muv_mean = np.zeros(data.shape[0])
    # muv_var = np.zeros(data.shape[0])
    log_density = np.zeros(input_data.shape[0])

    # dataset = tf.data.Dataset.from_tensor_slices(data)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float64)
    Z = tf.convert_to_tensor(Z.reshape(-1,1), dtype=tf.float64)
    # Z = Z.reshape(-1,1)
    dataset = tf.data.Dataset.from_tensor_slices((input_data, Z))
    # print(input_data.shape[0])
    batch_size = min(65000, input_data.shape[0])
    data_iter = iter(dataset.batch(batch_size))
    # raise Exception()
    i = 0 
    # prediction on the entire dataset at once results in OOM issues, needs to be done in batches :(
    for d in data_iter:
        log_density[i:i+batch_size] = model.predict_log_density(d).numpy()
        i+=batch_size
    # log_density = model.predict_log_density(data).numpy() 
    density = np.power(10.0,log_density)
    # density = density.reshape(nx,ny,nz).T # final shape is mag, z, lmh
    # mh = np.power(10.0,logmhs)
    # # print(d)
    # density = density / mh.reshape(1,1,-1)
    # density = density.T
    # density /= np.trapz(density, mh, axis=0)
    # return density.T #muv_mean.reshape(len(z_grid),-1), muv_var.reshape(len(z_grid),-1)
    return density.reshape(nx,ny,nz).T 


def get_data_pdf(observed_data, muv_grid, z_grid, pdf_fn, overwrite=False):
    # obs_pdf_fn = '/home/gdriskell/galacticus/data/ngdeep_pdf.npy'
    if path.isfile(pdf_fn) and not overwrite:
        obs_pdf = np.load(pdf_fn)
    else:
        nd = len(observed_data)
        nm = len(muv_grid)
        nz = len(z_grid)
        obs_pdf = np.zeros((nd,nm,nz))
        for i in range(nd):
            muv = observed_data['mf277w'][i]
            z = observed_data['z'][i]
            z_upper_err = observed_data['z_upper_err'][i]
            z_lower_err = np.abs(observed_data['z_lower_err'][i])
            dz = z_grid[1]-z_grid[0]
            muv_prob = np.zeros_like(muv_grid)
            idx = np.argmin(np.abs(muv_grid-muv))
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
        np.save(pdf_fn, obs_pdf)
    return obs_pdf


def compute_uvlf(weights, probs, data_dir, output_tag, recompute=False, abs_mag=False):
    # Output shape is muv x z
    if abs_mag:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf_abs.npy')
    else:
        uvlf_fn = path.join(data_dir, output_tag + '_uvlf.npy')
    if path.isfile(uvlf_fn) and not recompute:
        phi = np.load(uvlf_fn)
    else:
        print('weights shape ', weights.shape)
        print('probs shape ', probs.shape)
        # npad = weights.shape[-1]-probs.shape[-1]
        # probs = np.pad(probs, [[0,0],[0,0],[npad,0]])
        # print('new probs shape ', probs.shape)
        # probs = probs.T
        # probs = (probs / np.sum(probs, axis=0))
        # probs = probs.T

        norm = np.sum(probs, axis=0) # 0,1 looks most right?
        nmuv, nz, nmh = probs.shape
        norm = norm.reshape(1, nz, nmh)
        probs = (probs / norm)

        # print(probs[0,0,:][0:10], probs[0,0,:][-10:])

        # want to replace each prob by the median of the dist
        # np.amax(probs, axis=)
        # Mhs = np.geomspace(1.0e8,5.0e11,3699)[:-1]
        phi = np.sum(weights * probs, axis=2)
        # phi = np.trapz(weights*probs, Mhs, axis=2)
        np.save(uvlf_fn, phi)
    return phi


def plot_uvlf(muv_grid, z_grid, uvlf, volumes, data_dir):
    c = plt.cm.Dark2(3)
    zs = [8.0, 12.0, 16.0]
    f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)
    # for i,z in enumerate(zs):
    ax = axs[0]
    zidx = (z_grid > 8.5) & (z_grid < 9.5)
    # zidx = np.isclose(z_grid,8.0)
    y = np.sum(uvlf[:,zidx]*volumes[zidx],axis=1)
    y /= np.sum(volumes[zidx])
    # y = np.average(uvlf[:, zidx],axis=1) 

    # print(np.amin(y), np.amax(y))
    midx = y>0
    y = np.log10(y[midx])
    ax.plot(muv_grid[midx], y, label='Sim.', lw=2.5, c=c)
    ngdeep_muv = [-20.1, -19.1, -18.35, -17.85, -17.35]
    ngdeep_phi = np.array([14.7e-5, 18.9e-5, 74.0e-5, 170.0e-5, 519.0e-5])
    ngdeep_phi_err = np.array([[7.2e-5, 8.9e-5, 29.0e-5, 65.e-5, 198.e-5],
                      [11.1e-5, 13.8e-5, 41.4e-5, 85.e-5, 248.e-5]])
    obs_log_phi = np.log10(ngdeep_phi)
    log_err = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-obs_log_phi]
    
    # donnan_muv = [-22.30, -21.30, -18.50]
    # donnan_phi = np.array([0.17e-6, 3.02e-6, 1200e-6])
    # donnan_log_phi = np.log10(donnan_phi)
    # donnan_upper_err = [0.4e-6, 3.98e-6, 717e-6]
    # donnan_lower_err = [0.14e-6, 1.95e-6, 476e-6]
    # donnan_err = np.array([donnan_lower_err, donnan_upper_err])
    # donnan_log_err = [donnan_log_phi-np.log10(donnan_phi-donnan_err[0,:]),np.log10(donnan_phi+donnan_err[1,:])-donnan_log_phi]
    
    # neglecting the 0 bins
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
    zidx = (z_grid > 9.5) & (z_grid < 12.0)
    # zidx = np.isclose(z_grid,12.0)
    y = np.sum(uvlf[:,zidx]*volumes[zidx],axis=1)
    y /= np.sum(volumes[zidx])
    # y = np.average(uvlf[:, zidx],axis=1) 
    midx = y>0
    y = np.log10(y[midx])
    ax.plot(muv_grid[midx], y, label='Sim.', lw=2.5, c=c)
    ngdeep_muv = [-19.35, -18.65, -17.95, -17.25]
    ngdeep_phi = np.array([18.5e-5, 27.7e-5, 59.1e-5, 269.0e-5])
    ngdeep_phi_err = np.array([[8.3e-5, 13.0e-5, 29.3e-5, 124.e-5],
                    [11.9e-5, 18.3e-5, 41.9e-5, 166.e-5]])
    obs_log_phi = np.log10(ngdeep_phi)
    log_err = [obs_log_phi-np.log10(ngdeep_phi-ngdeep_phi_err[0,:]),np.log10(ngdeep_phi+ngdeep_phi_err[1,:])-obs_log_phi]

    # donnan_muv = [-22.57, -20.10, -19.35, -18.85, -18.23]
    # donnan_phi = np.array([0.18e-6, 16.2e-6, 136.0e-6, 234.9e-6, 630.8e-6])
    # donnan_log_phi = np.log10(donnan_phi)
    # donnan_upper_err = [0.42e-6, 21.4e-6, 67.2e-6, 107e-6, 340e-6]
    # donnan_lower_err = [0.15e-6, 10.5e-6, 47.1e-6, 76.8e-6, 233e-6]
    # donnan_err = np.array([donnan_lower_err, donnan_upper_err])
    # donnan_log_err = [donnan_log_phi-np.log10(donnan_phi-donnan_err[0,:]),np.log10(donnan_phi+donnan_err[1,:])-donnan_log_phi]
    ceers_muv_z11 = [-20.5, -20.0, -19.5, -19.0, -18.5]
    ceers_phi_z11 = np.array([1.8e-5, 5.4e-5, 7.6e-5, 17.6e-5, 26.3e-5])
    ceers_log_phi = np.log10(ceers_phi_z11)
    ceers_upper_z11 = np.array([1.2e-5, 2.7e-5, 3.9e-5, 10.3e-5, 18.2e-5])
    ceers_lower_z11 = np.array([0.9e-5, 2.1e-5, 3.0e-5, 7.9e-5, 13.3e-5])
    ceers_log_err = [ceers_log_phi-np.log10(ceers_phi_z11-ceers_lower_z11),np.log10(ceers_phi_z11+ceers_upper_z11)-ceers_log_phi]
    

    ax.scatter(ngdeep_muv, obs_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='k')
    ax.errorbar(ngdeep_muv, obs_log_phi, yerr=log_err, elinewidth=1.8, label='NGDEEP data (Leung et. al. 2023)', marker='o', color="none", ecolor='k', ls='none', capsize=3.5)
    #ax.scatter(donnan_muv, donnan_log_phi, marker='o', facecolor="none", edgecolor='gray', label='Donnan 2022')
    #ax.errorbar(donnan_muv, donnan_log_phi, yerr=donnan_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.scatter(ceers_muv_z11, ceers_log_phi, linewidths=1.5, marker='o', facecolor="none", edgecolor='gray', label='CEERS (Finkelstein et. al. 2023)')
    ax.errorbar(ceers_muv_z11, ceers_log_phi, elinewidth=1.8, yerr=ceers_log_err, marker='o', color="none", ecolor='gray', ls='none', capsize=3.5)
    ax.set_title(r'$9.5<z<12.0$')
    ax.set_xlabel(r'$M_{\mathrm{UV}}$')
    ax.set_ylabel(r'$\mathrm{Log}\left(\phi_{\mathrm{UV}}\right)$')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(-22.5, -16.5)
    ax.set_ylim(-6, -1.5)
    plt.savefig(path.join(data_dir, 'model_uvlf.png'), dpi=600)
    plt.close('all')
    

def compare_binned_uvlf(X, Y, data_dir, mag_grid, weights, z_grid, uvlf, output_tag):
    logmhs = np.unique(X[:,0])
    lmh_bins = np.linspace(8, 11, 31)
    dlmh = lmh_bins[1]-lmh_bins[0]
    zs = [8.0, 12.0, 16.0]
    f, axs = plt.subplots(len(zs), 1, figsize=(6,15),constrained_layout=True)
    dmuv = 0.5
    min_mag = 26.5
    max_mag = 32.5
    muv_bins = np.arange(min_mag, max_mag+0.1, dmuv) + 0.05
    Y = np.where(Y>32.0, 32.1, Y)
    nbins = len(muv_bins)-1
    hist_samples = 200
    for i,z in enumerate(zs):
        ax = axs[i]
        idx = X[:,1] == z
        Yi = Y[idx]
        Xi = X[idx]
        raw_data_uvlf = np.zeros(nbins)
        data_uvlf = np.zeros(nbins)
        model_uvlf = np.zeros(nbins)
        z_idx = z_grid == z
        uvlf_z = uvlf[:,z_idx] 
        for j in range(nbins):
            left = muv_bins[j]-dmuv/2.0 
            right = muv_bins[j]+dmuv/2.0 
            mh_idx = (Yi>left)&(Yi<right)
            idx_weights = weights[z_idx,mh_idx]
            raw_data_uvlf[j] = np.sum(idx_weights)
            

            for k,lmh in enumerate(lmh_bins[:-1]):
                left_lmh = lmh - dlmh/2.0
                right_lmh = lmh + dlmh/2.0
                lidx = np.argmin(np.abs(Xi[:,0]-left_lmh))
                ridx = np.argmin(np.abs(Xi[:,0]-right_lmh))
                
                plot_x = Xi[lidx:ridx,0]
                plot_Y = Yi[lidx:ridx]
                # print(k, left_lmh, right_lmh)
                # print(np.amin(plot_Y), np.amax(muv_bins))
                if np.amin(plot_Y) > (np.amax(muv_bins)-(dmuv/2.0)):
                    # print(f'skipping lmh = {lmh}')
                    continue
                else:                    
                    hist, _ = np.histogram(plot_Y, bins = muv_bins-(dmuv/2.0)) # P(Muv|Mh)
                    if np.sum(hist) < 1:
                        continue
                    hist = hist.astype(float)
                    hist /= np.sum(hist)
                    # print(len(hist))
                    data_uvlf[j] += np.sum(hist[j]*weights[z_idx,lidx:ridx])

            # print(Xi[mh_idx,0].shape, idx_weights.shape)
            # hist_weights = idx_weights / 
            # prob_weight,_ = np.histogram(Xi[mh_idx,0], weights=idx_weights, bins=lmh_bins, density=True)
            # print(prob_weight.shape)
            # prob_weight /= np.sum(prob_weight) # should already be normalized but being safe
            # print(lmh_bins.shape)
            # for k,lmh in enumerate(lmh_bins[:-1]):
            #     closest_idx = np.argmin(np.abs(logmhs - (lmh + dlmh/2.0)))
            #     data_uvlf[j] += prob_weight[k] * weights[z_idx, closest_idx]
            # data_uvlf[j] = np.sum(prob_weight * idx_weights)

            mag_idx = (mag_grid > left) & (mag_grid < right)
            model_uvlf[j] = np.sum(uvlf_z[mag_idx])
        raw_data_uvlf /= dmuv
        data_uvlf /= dmuv
        model_uvlf /= dmuv
        ax.set_title(f'$z={z}$')
        ax.semilogy(muv_bins[:-2], raw_data_uvlf[:-1], label='Binned UVLF')
        ax.semilogy(muv_bins[:-2], data_uvlf[:-1], '--', label='Hist. weighted binned UVLF')
        ax.semilogy(muv_bins[:-2], model_uvlf[:-1], ':', label='Model UVLF')
        ax.legend(frameon=False)
        ax.set_xlim(26.5, 30.5)
        ax.set_xlabel(r'$m_{\mathrm{uv}}$')
        ax.set_ylabel(r'$\phi_{\mathrm{uv}} \mathrm{[Mpc^{-1} mag^{-1}]}$')
    plt.savefig(path.join(data_dir, output_tag+'_uvlf_comp.pdf'))
    plt.close('all')
    

# def plot_ngal_vs_z(muv_grid, z_grid, uvlf, data_dir):
#     zs = [8.0, 12.0, 16.0]
#     f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)
#     # for i,z in enumerate(zs):
#     ax = axs[0]
#     zidx = (z_grid > 8.5) & (z_grid < 9.5)
#     # print(np.sum(zidx))
#     y = np.sum(uvlf[:, zidx],axis=1)
#     # print(np.amin(y), np.amax(y))
#     midx = y>0
#     y = np.log10(y[midx])
#     ax.plot(muv_grid[midx], y, label='Sim.')


def plot_mh_given_muv(logmhs, probs, weights, mh_cutoff, data_dir, output_tag='', X=None,Y=None):
    # Probs shape: muv x zs x mhs
    # weights shape: zs x mhs
    # if not 
    logmhs = np.unique(logmhs)

    mh_probs = weights / np.sum(weights, axis=-1).reshape(-1, 1)

    npad = weights.shape[-1]-probs.shape[-1]
    probs = np.pad(probs, [[0,0],[0,0],[npad,0]])
    muv_prob = np.sum(mh_probs * probs, axis=2) # muv x z
    prob_mh_given_muv = mh_probs * probs / muv_prob.reshape(*muv_prob.shape, 1)

    # zs = [16.0, 12.0, 8.0]
    zs = [16.0]
    muvs = [-18, -19, -20]
    # f, axs = plt.subplots(1, len(zs), figsize=(16,5),constrained_layout=True)
    f,ax = plt.subplots()
    
    # plot_muvs = np.linspace(27, 31, 50).reshape(50,1)
    # plot_muv_min = 27
    # plot_muv_max = 31
    # midx = np.where((app_grid > plot_muv_min) & (app_grid < plot_muv_max))

    for i,z in enumerate(zs):
        # ax = axs[i]
        # print(X[:,-1])
        zidx = z_grid==z
        colors = iter([plt.cm.Dark2(i) for i in range(8)])
        for muv in muvs:
            color = next(colors)
            midx = abs_grid == muv
            pmz = prob_mh_given_muv[midx,zidx,:]
            # print(pmz.shape)
            ax.fill_between(logmhs.flatten(), pmz.flatten()/np.amax(pmz), color=color, alpha=0.5, label=r'$M_{\mathrm{UV}}='+str(muv)+'$')

            # if not (X is None):
            #     # where X =
            #     probs = np.hist(Y_data, bins=muv_grid)
        ax.set_ylabel(r'$P\left(M_{h}|M_{\mathrm{UV}}\right)$')
        ax.legend(frameon=False, fontsize=12.5)
        # ax.set_title(f'$z={z}$')
        ax.set_xlim(8.9, 10.6)
        ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
    plt.tight_layout()
    # plt.savefig('Prob_Mh_'+output_tag+'.png', dpi=600)
    plt.savefig('Prob_Mh.pdf')
    plt.close('all')

    # f, axs = plt.subplots(len(zs), 1, figsize=(6,15),constrained_layout=True, sharex=True)
    # for i,z in enumerate(zs):
    #     ax = axs[i]
    #     # print(X[:,-1])
    #     zidx = z_grid==z
    #     colors = iter([plt.cm.Dark2(i) for i in range(8)])
    #     # mv = 0
    #     for muv in muvs:
    #         color = next(colors)
    #         midx = abs_grid == muv
    #         pmz = prob_mh_given_muv[midx,zidx,:]*weights[zidx,:]
    #         # print(pmz.shape)
    #         ax.fill_between(logmhs.flatten(), pmz.flatten()/np.amax(pmz), color=color, alpha=0.5, label=r'$M_{\mathrm{UV}}='+str(muv)+'$')
    #         # mv = max(mv, np.amax(pmz))
    #     ax.set_ylabel(r'$P\left(M_{h}|M_{\mathrm{UV}}\right)\times \mathrm{n(M)}$')
    #     ax.legend(frameon=False, fontsize=12.5)
    #     ax.set_title(f'$z={z}$')
    #     ax.set_yscale('log')
    #     ax.set_ylim(1e-2, 1)
    # ax.set_xlabel(r'$\mathrm{Log}\left(M_{h}\right)$')
    # plt.xlim(8.0, 11)
    # plt.savefig(path.join(data_dir, 'weighted_mh_dist.pdf'))
    # plt.close('all')


def get_ngdeep_cf(app_grid):
    """z~9 covers z=8.5-9.5 and z~11 covers z=9.5-12.0
    Can't find survey area so just dividing by max for now"""
    z9_volume = cosmo.comoving_volume(9.5)-cosmo.comoving_volume(8.5)
    z11_volume = cosmo.comoving_volume(12.0)-cosmo.comoving_volume(9.5)
    ngdeep_absmag_z9 = np.array([-21.1, -20.1, -19.1, -18.35, -17.85, -17.35])
    ngdeep_appmag_z9 = ngdeep_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
    ngdeep_veff_z9 = np.array([18700., 18500., 15800., 13100., 7770., 2520.]) # Mpc^3
    ngdeep_absmag_z11 = np.array([-20.05, -19.35, -18.65, -17.95, -17.25, -17.05]) # last value using extrapolation
    ngdeep_veff_z11 = np.array([27900., 26100., 20800., 9840., 2210., 0.]) 
  
    ngdeep_appmag_z11 = ngdeep_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
    ngdeep_z9_cf = ngdeep_veff_z9 / np.amax(ngdeep_veff_z9) #z9_volume
    ngdeep_z11_cf = ngdeep_veff_z11 / np.amax(ngdeep_veff_z11) #z11_volume

    ### TODO: really dumb version for now <------!
    ngd_cf = np.concatenate(([1.0], ngdeep_z11_cf, [0.0]))
    ngd_app = np.concatenate(([np.amin(app_grid)], ngdeep_appmag_z11, [np.amax(app_grid)]))
    cf = np.interp(app_grid, ngd_app, ngd_cf)
    # print(cf)
    return cf


def get_ceers_cf(app_grid):
    """z~9 covers z=8.5-9.7 and z~11 covers z=9.7-13.0, z~14 covers > 13
    Can't find survey area so just dividing by max for now"""
    z9_volume = cosmo.comoving_volume(9.7)-cosmo.comoving_volume(8.5)
    z11_volume = cosmo.comoving_volume(13.0)-cosmo.comoving_volume(9.7)
    z14_volume = cosmo.comoving_volume(14.5)-cosmo.comoving_volume(13.0)
    ceers_absmag_z9 = np.array([-22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5, -18.0])
    ceers_appmag_z9 = ceers_absmag_z9+cosmo.distmod(9.0).value-2.5*np.log10(1+9.0)
    ceers_veff_z9 = np.array([187000., 187000., 187000., 193000., 177000., 161000., 120000., 77900., 18600., 0.0]) # Mpc^3
    
    ceers_absmag_z11 = np.array([-21.0, -20.5, -20.0, -19.5, -19.0, -18.5, -18.0]) # last value using extrapolation
    ceers_veff_z11 = np.array([373000., 354000., 306000., 220000., 62100., 23800., 0.0]) 
    ceers_appmag_z11 = ceers_absmag_z11+cosmo.distmod(11.0).value-2.5*np.log10(1+11.0)
    
    ceers_absmag_z14 = np.array([-20.5, -20.0, -19.5, -19.0,]) # last value using extrapolation
    ceers_veff_z14 = np.array([147000., 85400., 60600., 0.0]) 
    ceers_appmag_z14 = ceers_absmag_z11+cosmo.distmod(14.0).value-2.5*np.log10(1+14.0)

    ceers_z9_cf = ceers_veff_z9 / np.amax(ceers_veff_z9) #z9_volume
    ceers_z11_cf = ceers_veff_z11 / np.amax(ceers_veff_z11) #z11_volume
    ceers_z14_cf = ceers_veff_z14 / np.amax(ceers_veff_z14)

    
    ceers_cf = np.concatenate(([1.0], ceers_z11_cf, [0.0]))
    ceers_app = np.concatenate(([np.amin(app_grid)], ceers_appmag_z11, [np.amax(app_grid)]))
    cf = np.interp(app_grid, ceers_app, ceers_cf)
    # print(cf)
    return cf


def get_mh_cutoff(cutoff, X, mags):
    pivots = []
    nsample = 200
    threshold = 5
    for z in [8.0, 12.0, 16.0]:
        idx = X[:,-1]==z
        logmhs = X[idx,0]
        y = mags[idx]
        sorted_idx = np.argsort(logmhs)
        logmhs = logmhs[sorted_idx]
        y = y[sorted_idx]
        current = y[:nsample]
        # print(current)
        i=nsample
        while (i<(len(y)-nsample)) and (np.sum(current<cutoff)<threshold):
            current = y[i-nsample:i] 
            i+=1   
        pivots.append(logmhs[i])
    return np.amin(pivots)


ngdeep_data = pd.read_csv('/home/gdriskell/galacticus/data/ngdeep_data.csv')
ceers_data = pd.read_csv('/home/gdriskell/galacticus/data/ngdeep_data.csv')
zgrid_weights = get_weights_from_hmf('/home/gdriskell/galacticus/data/gpr_zgrid_hmfs:MPI0000.hdf5')

# TODO: THIS IS JUST TO TEST
# zgrid_weights = (zgrid_weights[:,:-1]+zgrid_weights[:,1:])/2.0
zgrid_weights = zgrid_weights[:,:-1]

z_grid = np.linspace(7.0, 17.0, 21)
dz = z_grid[1]-z_grid[0]

abs_min = -25.0
abs_max = -12.0
nabs = int(round((abs_max-abs_min)/0.25))+1
abs_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = abs_grid[1]-abs_grid[0]

app_min = 23.0
app_max = 35.0
napp = int(round((app_max-app_min)/0.25))+1
app_grid = np.linspace(app_min, app_max, napp) 
dapp = app_grid[1]-app_grid[0]
# print(len(app_grid))

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
ngdeep_pdf = np.abs(get_data_pdf(ngdeep_data, app_grid, z_grid, ngdeep_pdf_fn, False))
ceers_pdf = np.abs(get_data_pdf(ceers_data, app_grid, z_grid, ceers_pdf_fn, False))


def evaluate_likelihood(app_mags, abs_mags, weights, logmhs, zs, data_dir, plot_abs=False, task='load', output_tag='', compute_likelihood=False):
    abs_fn = path.join(data_dir, 'abs_model_output.pdf')
    abs_uvlf_fn = path.join(data_dir, output_tag + '_uvlf_abs.npy')
    
    if plot_abs:
        # if task=='load' and path.isfile(abs_uvlf_fn):
        if False:
            uvlf = compute_uvlf(None, None, data_dir, output_tag, False, True)/dabs
            plot_uvlf(abs_grid, z_grid, uvlf, data_dir)
        else:
            X_train = np.hstack((logmhs.reshape(-1,1),zs.reshape(-1,1)))
            # mh_cutoff = get_mh_cutoff(-14.0, X_train, abs_mags)
            # abs_idx = logmhs # >mh_cutoff 
            # X_train_abs = X_train[abs_idx,:]
            # abs_mags = abs_mags[abs_idx]
            # print(len(np.uniques(logmhs)))
            model = get_sparse_model(X_train, abs_mags, data_dir, True, task, output_tag)

            # probs = get_stats_from_model(model, logmhs[abs_idx], z_grid, abs_grid)
            probs_fn = path.join(data_dir, output_tag +'_abs_probs.npy')
            if path.isfile(probs_fn) and not ((task=='retrain') or (task=='continue')):
                probs = np.load(probs_fn)
            else:
                probs = get_stats_from_model(model, logmhs, z_grid, abs_grid)
                np.save(probs_fn, probs)
            # plot_model_output(mh_cutoff, X_train_abs, abs_mags, model, data_dir, abs_grid, True, output_tag)
            mh_cutoff = np.amin(logmhs)
            plot_model_output(mh_cutoff, X_train, abs_mags, model, data_dir, abs_grid, True, output_tag)
            plot_interp_output(mh_cutoff, model, data_dir, abs_grid, True, output_tag)

            # plot_mh_given_muv(logmhs, probs, weights, mh_cutoff, data_dir) 
            # weight_mhs = (weight_mhs[:-1]+weight_mhs[1:])/2.0
            # widx = np.log10(weight_mhs) > mh_cutoff
            uvlf = compute_uvlf(weights, probs, data_dir, output_tag, True, True)/dabs
            # return None
            plot_uvlf(abs_grid, z_grid, uvlf, z_volumes, data_dir)
            
        # plot_model_output(mh_cutoff, X_train_abs, abs_mags, model, data_dir, abs_grid, True, output_tag)
        # plot_data_hist(mh_cutoff, X_train_abs, abs_mags, model, data_dir, abs_grid, True, output_tag)
        # plot_model_and_weights(mh_cutoff,  X_train_abs, abs_mags, model, data_dir, weights, True)
        # plot_mh_given_muv(logmhs, probs, weights, mh_cutoff, data_dir)
    return 
    uvlf_fn = path.join(data_dir, output_tag + '_uvlf.npy')
    
    if task=='load' and path.isfile(uvlf_fn):
        print('loading uvlf')
        uvlf = compute_uvlf(None, None, data_dir, output_tag) 
    else:
        X_train = np.hstack((logmhs.reshape(-1,1),zs.reshape(-1,1)))
        mh_cutoff = get_mh_cutoff(31.5, X_train, app_mags)
        idx = logmhs>mh_cutoff 
        X_train = X_train[idx,:]
        app_mags = app_mags[idx]
        model = get_sparse_model(X_train, app_mags, data_dir, False, task, output_tag)
        plot_model_output(mh_cutoff, X_train, app_mags, model, data_dir, app_grid, False, output_tag)
        # plot_data_hist(mh_cutoff, X_train, app_mags, model, data_dir, app_grid, False, output_tag)
        # test_model_diff(X_train, app_mags, data_dir, app_grid, weights, model, z_grid, output_tag)
        probs = get_stats_from_model(model, logmhs[idx], z_grid, app_grid)
        uvlf = compute_uvlf(weights, probs, data_dir, output_tag, False, False) 

    if compute_likelihood:      
        z_cutoff = 8.5
        zidx = z_grid > z_cutoff
        ngdeep_n = np.sum(uvlf[:,zidx]*ngdeep_eff_volume[:,zidx], dtype=np.longdouble) 
        ceers_n = np.sum(uvlf[:,zidx]*ceers_eff_volume[:,zidx], dtype=np.longdouble) 
        print(f'ngdeep total = {ngdeep_n}, ceers total = {ceers_n}')
        loglike = -ngdeep_n-ceers_n
        ngdeep_obs_n = np.sum(ngdeep_pdf[:,:,zidx]*uvlf[:,zidx]*ngdeep_eff_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
        ngdeep_obs_n = np.log(ngdeep_obs_n[ngdeep_obs_n>0])
        ngdeep_obs_n = np.sum(ngdeep_obs_n)
        loglike += ngdeep_obs_n
        ceers_obs_n = np.sum(ceers_pdf[:,:,zidx]*uvlf[:,zidx]*ceers_eff_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
        ceers_obs_n = np.log(ceers_obs_n[ceers_obs_n>0])
        ceers_obs_n = np.sum(ceers_obs_n)
        loglike += ceers_obs_n
        print(f'loglike={loglike}')
    else: 
        loglike = None
    return loglike


def load_data(data_dir, reload=False):
    data_fn = path.join(data_dir, 'data.npy')
    if path.isfile(data_fn) and not reload:
        data = np.load(data_fn)
    else:
        app_mags = []
        abs_mags = []
        logmhs = []
        zs = []
        for z in ['8.0', '12.0', '16.0']: #
            print(z)
            #outfiles = [data_dir+f'z{z}:MPI{i:04d}.hdf5' for i in range(160)] #glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            outfiles = glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            outfiles += glob.glob(data_dir+f'lowMh_z{z}:MPI*.hdf5')
            mhs = []
            app = []
            abs = []
            weights = []
            for fn in outfiles:
                try: 
                    f = h5py.File(fn,"r")
                    if 'Outputs' in f:
                        d,w = get_Muv_from_hdf5(f, 'JWST_NIRCAM_f277w')
                        mhs.append(d[:,0].flatten())
                        app.append(d[:,1].flatten())
                        abs.append(d[:,2].flatten())
                        # print(len(mhs[-1]))
                    f.close()
                except:
                    continue
            mhs = np.concatenate(mhs)
            idx = np.argsort(mhs)
            mhs = mhs[idx]
            app = np.concatenate(app)[idx]
            abs = np.concatenate(abs)[idx]
            logmhs.append(mhs)
            app_mags.append(app)
            abs_mags.append(abs)
            zs.extend([float(z) for i in range(len(logmhs[-1]))])
        
        app_mags = np.concatenate(app_mags)
        abs_mags = np.concatenate(abs_mags)
        logmhs = np.concatenate(logmhs)
        zs = np.array(zs)
        data = np.stack([logmhs, zs, app_mags, abs_mags], axis=1)
        np.save(data_fn, data)
    return data


def get_astro_params(data_dir):
    Vout_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
    alphaOut_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
    tau0_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/timescale'
    alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/exponentVelocity'
    xml_fn = path.join(data_dir,'z8.0.xml')
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    Vout = root.find(Vout_xml).get('value')
    alphaOut = root.find(alphaOut_xml).get('value')
    tau0 = root.find(tau0_xml).get('value')
    alphaStar = root.find(alphaStar_xml).get('value')
    return float(Vout), float(alphaOut), float(tau0), float(alphaStar)
    

def run(dirname, i, task, tag, reload, like):
    print(f'Starting p{i}')
    base = '/data001/gdriskell/jwst_blitz_astro_samples/'
    data_dir = path.join(base, dirname+f'_p{i}/')
    logmhs, zs, app_mags, abs_mags = load_data(data_dir, reload).T
    loglike = evaluate_likelihood(app_mags, abs_mags, zgrid_weights, logmhs, zs, data_dir, plot_abs, task, tag, like)
    Vout, alphaOut, tau0, alphaStar = get_astro_params(data_dir)
    return Vout, alphaOut, tau0, alphaStar, loglike


plot_abs = True


if __name__ == "__main__":
    parser = ArgumentParser(description="")

    parser.add_argument("dirname", help="Path to data directory")
    parser.add_argument("--initial", type=int, help="Initial index to run")
    parser.add_argument("--final", type=int, help="Final index to run")
    parser.add_argument("--task", help="Task for GP (retrain, contiune, or load)")
    parser.add_argument("--tag", default='', help="Tag for plot labels")
    parser.add_argument("--save", action='store_true', help="Whether to save output to file")
    parser.add_argument("--reload", action='store_true', help="Whether to reload data")
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
        Vout, alphaOut, tau0, alphaStar, loglike = run(args.dirname, i, args.task, args.tag, args.reload, args.like)
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
        df.to_csv('updated_params.csv')
