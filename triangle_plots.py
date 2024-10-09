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
import cmasher as cmr
import seaborn as sns
# from sklearn.neighbors import KernelDensity

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


def get_Muv_from_hdf5(outfile, temp=False):
    outputs = outfile['Outputs']    
    nodeData = outputs['Output1']['nodeData']
    Mhs = nodeData['basicMass'][:]
    # mhidx = Mhs > 1e8
    Mhs = Mhs # changed
    if temp:
        treeWeights = outputs['Output1/mergerTreeWeight'][:]
    else:
        treeWeights = nodeData['mergerTreeWeight'][:]

    z = nodeData['redshift'][0]
    sfn = 'spheroidLuminositiesStellar:JWST_NIRCAM_f200w:observed:z{:.4f}'.format(z)
    dfn = 'diskLuminositiesStellar:JWST_NIRCAM_f200w:observed:z{:.4f}'.format(z)
    Lum = nodeData[sfn][:]+nodeData[dfn][:] # changed
    # print(np.sum(np.isnan(Lum)), np.sum(Lum==0.0))
    # print(np.amax(Mhs[Lum<1e-2]))
    Lum[Lum<1.0] = 1.0 # np.amin(Lum>0.0)

    # print(np.amin(Lum))
    M_uvs = -2.5*np.log10(Lum)
    sortIdx = np.argsort(Mhs)
    Mhs = Mhs[sortIdx]
    M_uvs = M_uvs[sortIdx]
    treeWeights = treeWeights[sortIdx]
    data = np.stack((np.log10(Mhs), M_uvs),axis=-1)
    return data, treeWeights


def plot_astro_pMuv(fn_root, df, parameters, z=None, plot_filled=False):
    labels = parameters_to_labels(parameters)
    if z is None:
        pMuv = np.abs(df['pMuv'].values)
    else:
        pMuv = np.abs(df['pMuv_z{}'.format(z)].values)
    # print('Most likely is {}'.format(np.amax(pMuv)))
    # pMuv /= np.amax(pMuv)
    nparam = len(parameters)
    uniques = [np.unique(df[p]) for p in parameters]

    tab_prob_1d = tabulate_1d(df, uniques, parameters, pMuv)
    like_prob_1d = tabulate_1d(df, uniques, parameters, np.abs(df['like'].values))
    tab_prob_2d, prob_norm = tabulate_2d(df, uniques, parameters, pMuv)

    
    vmin = max(np.amin(pMuv), 1e-10)
    norm = mpl.colors.LogNorm(vmin=1e-3, vmax=np.amax(pMuv))
    # norm = mpl.colors.LogNorm(vmin=1e-2, vmax=1)
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = 'viridis'
    frac = 0.10
    f,axs = plt.subplots(nparam, nparam, figsize=(10*(1+frac),10), sharex='col')
    for i,pi in enumerate(parameters):
        for j,pj in enumerate(parameters):
            xs = []
            ys = []
            probs = []
            ax = axs[i,j]
            label_axis(axs,i,j,labels,uniques, parameters)
            if j<(i+1):
                if i==j:
                    for vi in uniques[i]:
                        idx = np.isclose(df[pi], vi)
                        if np.any(idx):
                            xs.append(vi)
                            # probs.append(np.mean(like[idx]))
                    # probs /= np.trapz(probs, xs)
                    prob = np.array(tab_prob_1d[i][:])
                    prob /= np.amax(prob)
                    ax.plot(xs, prob, label=r'$P(M_{\mathrm{UV}}^{\mathrm{obs}})$')
                    like = np.array(like_prob_1d[i][:])
                    like /= np.nanmax(like)
                    ax.plot(xs, like, label='Full Likelihood')
                    if i ==0:
                        ax.legend(frameon=False, bbox_to_anchor=(1.9, 1.0))
                    ax.set_ylim(-0.05,1.05)
                    # ax.yaxis.tick_right()
                    # ax.set_yticks([0, 0.5, 1])
                    # ax.set_yticklabels(['0.0','0.5', '1.0'])
                    axs[i,j].get_yaxis().set_visible(False)
                if i!=j:
                    for vi in uniques[i]:
                        for vj in uniques[j]:
                            idx = (df[pi] == vi) & (df[pj] == vj)
                            if np.any(idx):
                                xs.append(vj)
                                ys.append(vi)
                                #  = like[idx]
                                # prob2d = np.mean(like[idx])
                                # probs.append(prob2d)
                    prob = np.array(tab_prob_2d[i][j])#/prob_norm
                    prob /= np.nanmax(prob)
                    if plot_filled:
                        # xs = np.array(xs).reshape(prob.shape)
                        # ys = np.array(ys).reshape(prob.shape)
                        x = get_flat_data_points(pj, uniques[j])
                        y = get_flat_data_points(pi, uniques[i])
                        xx,yy = np.meshgrid(x,y)
                        # print(xs, ys)
                        ax.pcolormesh(xx, yy, prob, 
                                           cmap = cmap, norm = norm, shading = 'flat')
                        ax.scatter(xs,ys,c=prob.flatten(), marker='o', cmap=cmap, norm=norm,
                                        edgecolor='k')
                        # im = ax.imshow(prob, cmap=cmap, norm=norm,
                        #                extent=(xs.min(), xs.max(), ys.max(), ys.min()),
                        #                interpolation='nearest', origin='lower', aspect='auto')
                        
                    else:
                        im = ax.scatter(xs,ys, c=prob.flatten(), cmap=cmap, norm=norm)
                    # if (pi == 'velocityOutflow') or (p == 'starFormationFrequencyNormalization'):
            else:
                ax.axis('off')
    # plt.minorticks_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pad = 0.01
    cbar = f.colorbar(sm, ax=axs[-1,:], pad=pad, extend='min',
                      fraction=frac)
    # cbar.set_ticks([1e-10, 1e-5, 1])
    # cbar.set_ticklabels([r'$<10^{-10}$', r'$10^{-5}$', r'$1$'])
    # cbar.set_ticks([1e-2, 1e-1, 1])
    # cbar.set_ticklabels([r'$<10^{-2}$' ,r'$10^{-1}$', r'$1$'])
    # cbar.ax.set_yticklabels([r'$<0.2$', r'$2$', r'$>20$']) 
    f.colorbar(sm, ax=axs[0,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[1,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[2,:], pad=pad, fraction=frac).ax.set_visible(False)
    #plt.savefig('plots/prob_plots/astro_triangle_pMuv.pdf', bbox_inches='tight')
    if z is None:
        plt.savefig('plots/prob_plots/'+fn_root+'_triangle_pMuv.pdf', bbox_inches='tight')
    else:
        plt.savefig('plots/prob_plots/'+fn_root+'_triangle_pMuv_z{}.pdf'.format(z), 
            bbox_inches='tight')
    plt.close('all')


def plot_astro_poisson(fn_root, df, parameters, z=None, plot_filled=False):
    labels = parameters_to_labels(parameters)
    if z is None:
        pMuv = np.abs(df['ntot'].values)
    else:
        pMuv = np.abs(df['ntot_z{}'.format(z)].values)
    # ntot = df['ntot'].values
    nobs=None
    poiss = poisson_prob(nobs, ntot)
    nparam = len(parameters)
    uniques = [np.unique(df[p]) for p in parameters]

    tab_prob_1d = tabulate_1d(df, uniques, parameters, poiss)
    like_prob_1d = tabulate_1d(df, uniques, parameters, np.abs(df['like'].values))
    tab_prob_2d, prob_norm = tabulate_2d(df, uniques, parameters, poiss)

    
    vmin = 1e-3
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=np.amax(poiss))
    # norm = mpl.colors.LogNorm(vmin=1e-2, vmax=1)
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = 'viridis'
    frac = 0.10
    height = 10
    f,axs = plt.subplots(nparam, nparam, figsize=(height*(1+frac),height), sharex='col')
    for i,pi in enumerate(parameters):
        for j,pj in enumerate(parameters):
            xs = []
            ys = []
            ns = []
            ax = axs[i,j]
            # if j==0:
            label_axis(axs,i,j,labels,uniques, parameters)
            if j<=i:
                if j==i:
                    for vi in uniques[i]:
                        idx = (df[pi] == vi)
                        if np.any(idx):
                            xs.append(vi)
                            # ns.append(np.mean(ntot[idx]))
                    # ax.plot(xs, ns)
                    prob = np.array(tab_prob_1d[i][:])
                    prob /= np.amax(prob)
                    ax.plot(xs, prob, label='Poisson')
                    like = np.array(like_prob_1d[i][:])
                    like /= np.amax(like)
                    ax.plot(xs, like, label='Full Likelihood')
                    if i ==0:
                        ax.legend(frameon=False, bbox_to_anchor=(1.9, 1.0))
                    ax.set_ylim(-0.05,1.05)

                    # ax.semilogy(xs, ns)

                    # ax.set_ylim(2e-2,2e2)
                    # ax.axhline(2, c='k', linestyle='--', zorder=-999)
                    # ax.yaxis.tick_right()
                    # ax.tick_params(axis='y', which='minor', right=False)
                    # ax.set_yticks([1e-1, 1e0, 1e1, 1e2])
                    # ax.set_yticklabels(['0.1','1', '10','100'])
                    ax.get_yaxis().set_visible(False)
                else:
                    for vi in uniques[i]:
                        for vj in uniques[j]:
                            idx = (df[pi] == vi) & (df[pj] == vj)
                            if np.any(idx):
                                xs.append(vj)
                                ys.append(vi)
                                # prob2d = np.mean(ntot[idx])
                                # ns.append(np.mean(ntot[idx]))
                    prob = np.array(tab_prob_2d[i][j])/prob_norm
                    if plot_filled:
                        # xs = np.array(xs).reshape(prob.shape)
                        # ys = np.array(ys).reshape(prob.shape)
                        x = get_flat_data_points(pj, uniques[j])
                        y = get_flat_data_points(pi, uniques[i])
                        xx,yy = np.meshgrid(x,y)
                        # print(xs, ys)
                        ax.pcolormesh(xx, yy, prob, 
                                           cmap = cmap, norm = norm, shading = 'flat')
                        ax.scatter(xs,ys,c=prob.flatten(), marker='o', cmap=cmap, norm=norm,
                                        edgecolor='k')
                        # im = ax.imshow(prob, cmap=cmap, norm=norm,
                        #                extent=(xs.min(), xs.max(), ys.max(), ys.min()),
                        #                interpolation='nearest', origin='lower', aspect='auto')
                        
                    else:
                        im = ax.scatter(xs,ys, c=prob.flatten(), cmap=cmap, norm=norm)
            else:
                ax.axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pad = 0.01
    cbar = f.colorbar(sm, ax=axs[-1,:], pad=pad, extend='min',
                      fraction=frac)
    # cbar.set_ticks([vmin, 1e-2, 1])
    # cbar.set_ticks([1e-2, 1e-1, 1])
    # cbar.set_ticklabels([r'$<10^{-2}$' ,r'$10^{-1}$', r'$1$'])
    # cbar.set_ticklabels([r'$<10^{-4}$', r'$10^{-2}$', r'$1$'])
    # cbar.ax.set_yticklabels([r'$<0.2$', r'$2$', r'$>20$']) 
    f.colorbar(sm, ax=axs[0,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[1,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[2,:], pad=pad, fraction=frac).ax.set_visible(False)
    
    plt.savefig('plots/prob_plots/'+fn_root+'_triangle_poisson.pdf', bbox_inches='tight')
    plt.close('all')


def parameters_to_labels(parameters):
    labels = []
    for p in parameters:
        if p == 'velocityOutflow':
            # tex = r'$\mathrm{Log}(V_{\mathrm{outflow}})$'
            tex = r'$V_{\mathrm{outflow}}\,\mathrm{[km/s]}$'
        elif p == 'alphaOutflow':
            # tex = r'$\mathrm{Log}(\alpha_{\mathrm{outflow}})$'
            tex = r'$\alpha_{\mathrm{outflow}}$'
        elif p == 'starFormationFrequencyNormalization':
            # tex = r'$\mathrm{Log}(\nu_{0,\mathrm{SF}})$'
            tex = r'$\nu_{0,\mathrm{SF}}$'
        elif p == 'surfaceDensityExponent':
            # tex = r'$\mathrm{Log}(\alpha_{\mathrm{HI}})$'
            tex = r'$\alpha_{\mathrm{HI}}$'
        elif p == 'efficiency':
            tex = r'$\epsilon_{\ast}$'
        elif p == 'timescale':
            tex = r'$\tau_{0}\,\mathrm{[Gyr]}$'
        elif p == 'alphaStar':
            tex = r'$\alpha_{\ast}$'
        else:
            raise Exception('Unknown Astro Parameter')
        labels.append(tex)
    return labels


def tabulate_2d(df, uniques, parameters, q):
    # q = df[qn].to_numpy()
    # q = q.to_numpy()
    data = np.array([df[p].to_numpy() for p in parameters]) 
    nparam = len(parameters)
    # q/=np.amax(q)
    # ptheta = 1/np.prod([(np.amax(x)-np.amin(x)) for x in uniques])
    total_idxs = list(range(nparam)) 
    prob2d = [] 
    postmax = 0
    for i in range(nparam):
        prob_i = []
        pidx_i = total_idxs[:]
        pidx_i.remove(i)
        for j in pidx_i:
            prob_ij = [] # list for each parameter
            pidx_j = pidx_i[:]
            pidx_j.remove(j)
            for vi in uniques[i]: 
                idxi = np.isclose(data[i], vi, atol=1e-10)
                pij = []
                for vj in uniques[j]:
                    idxij = (idxi & np.isclose(data[j], vj, atol=1e-10))
                    m,n = pidx_j
                    pijm = []
                    dm = []
                    for vm in uniques[m]:
                        idxijm = (idxij & np.isclose(data[m], vm, atol=1e-10)) 
                        vn = data[n][idxijm]
                        likeijmn = q[idxijm]
                        if len(vn)>1:
                            sorted_idx = np.argsort(vn)
                            likeijmn = likeijmn[sorted_idx]
                            dm.append(vm)
                            vn = vn[sorted_idx]
                            volume = np.amax(vn)-np.amin(vn) 
                            if volume==0:
                                print(i,j,m,n)
                                print(vi,vj,vm)
                                print(vn, volume)
                            # postijm = np.trapz(likeijmn, x=vn)/volume
                            numidx = np.logical_not(np.isnan(likeijmn))
                            # postijm = np.sum(likeijmn[numidx])
                            postijm = np.mean(likeijmn[numidx])
                            pijm.append(postijm)
                        elif len(vn)>0:
                            dm.append(vm)
                            vn = vn[0]
                            postijm = likeijmn[0]
                            pijm.append(postijm)
                    # print(dm)
                    if len(dm)>1:
                        # volume = np.amax(dm)-np.amin(dm)
                        # postij = np.trapz(pijm, x=dm)/volume
                        # postij = np.sum(pijm)
                        postij = np.mean(pijm)
                    elif len(dm)>0:
                        postij = pijm[0]
                    postmax = max(postmax,postij)
                    pij.append(postij)
                prob_ij.append(pij) 
            prob_i.append(prob_ij)
        prob2d.append(prob_i)
    
    return prob2d, postmax


# def tabulate_2d_nexpected(df, uniques, parameters, like,):
#     q = q.to_numpy()
#     data = np.array([df[p].to_numpy() for p in parameters]) 
#     nparam = len(parameters)
#     # q/=np.amax(q)
#     # ptheta = 1/np.prod([(np.amax(x)-np.amin(x)) for x in uniques])
#     total_idxs = list(range(nparam)) 
#     prob2d = [] 
#     postmax = 0
#     for i in range(nparam):
#         prob_i = []
#         pidx_i = total_idxs[:]
#         pidx_i.remove(i)
#         for j in pidx_i:
#             prob_ij = [] # list for each parameter
#             pidx_j = pidx_i[:]
#             pidx_j.remove(j)
#             for vi in uniques[i]: 
#                 idxi = np.isclose(data[i], vi, atol=1e-10)
#                 pij = []
#                 for vj in uniques[j]:
#                     idxij = (idxi & np.isclose(data[j], vj, atol=1e-10))
#                     m,n = pidx_j
#                     pijm = []
#                     dm = []
#                     for vm in uniques[m]:
#                         idxijm = (idxij & np.isclose(data[m], vm, atol=1e-10)) 
#                         vn = data[n][idxijm]
#                         likeijmn = q[idxijm]
#                         if len(vn)>1:
#                             sorted_idx = np.argsort(vn)
#                             likeijmn = likeijmn[sorted_idx]
#                             dm.append(vm)
#                             vn = vn[sorted_idx]
#                             volume = np.amax(vn)-np.amin(vn) 
#                             if volume==0:
#                                 print(i,j,m,n)
#                                 print(vi,vj,vm)
#                                 print(vn, volume)
#                             postijm = np.trapz(likeijmn, x=vn)/volume
#                             pijm.append(postijm)
#                         elif len(vn)>0:
#                             dm.append(vm)
#                             vn = vn[0]
#                             postijm = likeijmn[0]
#                             pijm.append(postijm)
#                     # print(dm)
#                     if len(dm)>1:
#                         volume = np.amax(dm)-np.amin(dm)
#                         postij = np.trapz(pijm, x=dm)/volume
#                     elif len(dm)>0:
#                         postij = pijm[0]
#                     postmax = max(postmax,postij)
#                     pij.append(postij)
#                 prob_ij.append(pij) 
#             prob_i.append(prob_ij)
#         prob2d.append(prob_i)
    
#     return prob2d, postmax



def tabulate_1d(df, uniques, parameters, q):
    # q = q.to_numpy()
    # q = df[qn].to_numpy()
    q/=np.amax(q)
    data = np.array([df[p].to_numpy() for p in parameters]) # rows are parameters
    nparam = len(parameters)
    # volumes = [(np.amax(x)-np.amin(x)) for x in uniques]
    total_idxs = list(range(nparam)) 
    prob1d = [] 
    for i in range(nparam):
        pi = []
        pidx_i = total_idxs[:]
        pidx_i.remove(i)
        for vi in uniques[i]: 
            idxi = np.isclose(data[i], vi, atol=1e-10)
            j,m,n = pidx_i
            pij = []
            for vj in uniques[j]:
                idxij = (idxi & np.isclose(data[j], vj, atol=1e-10))
                pijm = []
                dm = []
                for vm in uniques[m]:
                    idxijm = (idxij & np.isclose(data[m], vm, atol=1e-10))    
                    vn = data[n][idxijm]
                    sorted_idx = np.argsort(vn)
                    likeijmn = q[idxijm][sorted_idx]
                    if len(vn)>1:
                        dm.append(vm)
                        vn = vn[sorted_idx]
                        volume = np.amax(vn)-np.amin(vn) if len(vn)>1 else 1
                        # postijm = np.trapz(likeijmn, x=vn)/volume
                        numidx = np.logical_not(np.isnan(likeijmn))
                        # postijm = np.sum(likeijmn[numidx])
                        postijm = np.mean(likeijmn[numidx])
                        pijm.append(postijm)
                    elif len(vn)>0:
                        dm.append(vm)
                        vn = vn[0]
                        postijm = likeijmn[0]
                        pijm.append(postijm)
                if len(dm)>1:
                    volume = np.amax(dm)-np.amin(dm) if len(dm)>1 else 1
                    # postij = np.trapz(pijm, x=dm)/volume
                    # postij = np.sum(pijm)
                    postij = np.mean(pijm)
                elif len(dm)>0:
                    postij = pijm[0]
                pij.append(postij) 
            dj = uniques[j]
            # volume = np.amax(dj)-np.amin(dj) if len(dj)>1 else 1
            # posti = np.trapz(pij, x=dj)/volume
            # posti = np.sum(pij)
            posti = np.mean(pij)
            pi.append(posti) 
        prob1d.append(pi)
    
    return prob1d
 

def label_axis(axs,i,j,labels,uniques, parameters):
    ax = axs[i,j]
    pi = parameters[i]
    pj = parameters[j]
    if False: # pj in ['velocityOutflow', 'timescale', 'starFormationFrequencyNormalization', 'efficiency']:
        # print(pj)
        # ax.set_xscale('log')
        # ax.tick_params(axis='x', which='minor', bottom=False)
        pjmin,pjmax = np.amin(uniques[j]), np.amax(uniques[j])
        tick_min = np.ceil(np.log10(pjmin))
        tick_max = np.floor(np.log10(pjmax))
        nt = round(tick_max-tick_min)+1
        # if nt == 0:
        #     nt=1
        # ticks = np.logspace(tick_min, tick_max, nt)
        ticks = np.linspace(tick_min, tick_max, nt)
        ax.set_xticks(ticks)
    if False: #pi in ['velocityOutflow', 'timescale', 'starFormationFrequencyNormalization', 'efficiency']:
        # if j < i:
            # ax.set_yscale('log')
        # ax.tick_params(axis='y', which='minor', bottom=False)
        pimin,pimax = np.amin(uniques[i]), np.amax(uniques[i])
        tick_min = np.ceil(np.log10(pimin))
        tick_max = np.floor(np.log10(pimax))
        nt = round(tick_max-tick_min)+1
        # if nt == 0:
        #     nt=1
        # ticks = np.logspace(tick_min, tick_max, nt)
        ticks = np.linspace(tick_min, tick_max, nt)
        ax.set_yticks(ticks)
    

    # if (j==0) and (i>0):
    #     ax.set_xscale('log')
    #     ax.set_xticks([1e1, 1e2])
    #     ax.tick_params(axis='x', which='minor', bottom=False)
    # if not plot_timescale:
    #                 #print('not plotting timescale')
    #     if (j==2) and (i>2):
    #         ax.set_xscale('log')
    #         ax.set_xticks([1e-9, 1e-8])
    #         ax.tick_params(axis='x', which='minor', bottom=False)
    #     if (i==2) and (j<2):
    #         ax.set_yscale('log')
    #         ax.tick_params(axis='y', which='minor', left=False)
    # else:
    #     if (j==1) and (i>1):
    #         ax.set_xscale('log')
    #         # ax.set_xticks([1e-2, 1e-1, 1e0])
    #         ax.set_xticks([1e-3, 1e-1, 1e1])
    #         ax.tick_params(axis='x', which='minor', bottom=False)
    #     if (i==1) and (j<1):
    #         ax.set_yscale('log')
    #         ax.tick_params(axis='y', which='minor', left=False)
    if (j==0) and (i!=0):
        ax.get_shared_y_axes().join(ax, *axs[i,:i])
        ax.set_ylabel(labels[i])
        # ymin, ymax = np.amin(uniques[i]), np.amax(uniques[i])
        # dymin, dymax = np.abs(ymin*0.1), np.abs(ymax*0.1)
        # ax.set_ylim(ymin-dymin,ymax+dymax)
    elif j<i:
        ax.set_yticklabels([])
        ax.tick_params(left=True, right=True)
    if i==(nparam-1):
        # xmin, xmax = np.amin(uniques[j]), np.amax(uniques[j])
        # dxmin, dxmax = np.abs(xmin*0.1), np.abs(xmax*0.1)
        # ax.set_xlim(xmin-dxmin,xmax+dxmax)
        ax.set_xlabel(labels[j])


def get_flat_data_points(p,data):
    new_data = np.zeros(len(data)+1)
    
    if False: #p in ['velocityOutflow', 'timescale', 'starFormationFrequencyNormalization', 'efficiency']:
        if len(data) > 1:
            log_data = np.log10(data)
            delta = (log_data[1]-log_data[0])/2.0
            new_data[0] = 10.0**(log_data[0]-delta)
            new_data[1:] = 10.0**(log_data+delta)
        else: 
            print('DANGER: ONLY ONE DATA POINT PROVIDED')
            log_data = np.log10(data)
            delta = (np.log10(500.0)-np.log10(50.0))/3.0
            new_data[0] = 10.0**(log_data[0]-delta)
            new_data[1:] = 10.0**(log_data+delta)
    else:
        delta = (data[1]-data[0])/2.0
        new_data[0] = data[0]-delta
        new_data[1:] = data+delta
    # print(p, new_data)
    return new_data


def plot_1d(i, ax, values, tab_prob_1d, label):
    # pname = 
    print(label)
    values 
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


def plot_astro_like(fn_root, df, parameters,  z=None, plot_filled=False):
    labels = parameters_to_labels(parameters)
    if z is None:
        like = np.exp(df['loglike'].values)
        # like = df['loglike'].values
    else:
        like = np.abs(df['like_z{}'.format(z)].values)
    maxidx = np.argmax(like)
    # like /= np.amax(like)
    nparam = len(parameters)
    # uniques = [np.sort(np.unique(df[p])) for p in parameters]
    uniques = [np.sort(np.unique(df[p])) for p in parameters]
    print(uniques)
    print('Max likehood is {}'.format(like[maxidx]))
    print('MLE parameters are {},{},{},{}'.format(*[df[p].to_numpy()[maxidx] for p in parameters]))
    tab_prob_1d = tabulate_1d(df, uniques, parameters, like)
    tab_prob_2d, prob_norm = tabulate_2d(df, uniques, parameters, like)
    
    # vmin = 10**np.ceil(np.log10(np.amin(tab_prob_2d/prob_norm)))
    # print(vmin)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # norm = mpl.colors.LogNorm(vmin=1.e-4, vmax=1.)
    # cmap = 'binary'
    cmap = cmr.get_sub_cmap('cmr.neutral_r', 0.05, 1.0)
    frac = 0.10
    f,axs = plt.subplots(nparam, nparam, figsize=(11*(1+frac),11), sharex='col')
    for i,pi in enumerate(parameters):
        for j,pj in enumerate(parameters):
            xs = []
            ys = []
            probs = []
            ax = axs[i,j]
            label_axis(axs,i,j,labels,uniques,parameters)
            if j<(i+1):
                if i==j:
                    for vi in uniques[i]:
                        idx = np.isclose(df[pi], vi)
                        if np.any(idx):
                            xs.append(vi)
                            # probs.append(np.mean(like[idx]))
                    # probs /= np.trapz(probs, xs)
                    plot_1d(i, ax, xs, tab_prob_1d[i][:], labels[i])
                    # prob = np.array(tab_prob_1d[i][:])
                    # prob /= np.nanmax(prob)
                    # ax.plot(xs, prob)
                    # ax.set_ylim(-0.05,1.05)
                    # ax.yaxis.tick_right()
                    # ax.set_yticks([0, 0.5, 1])
                    # ax.set_yticklabels(['0.0','0.5', '1.0'])
                    # axs[i,j].get_yaxis().set_visible(False)
                if i!=j:
                    for vi in uniques[i]:
                        for vj in uniques[j]:
                            idx = (df[pi] == vi) & (df[pj] == vj)
                            if np.any(idx):
                                xs.append(vj)
                                ys.append(vi)
                                #  = like[idx]
                                # prob2d = np.mean(like[idx])
                                # probs.append(prob2d)
                    # print(tab_prob_2d[i][j])  
                    # print(len(uniques[i]))
                    # xs = np.array()
                    prob = np.array(tab_prob_2d[i][j])#/prob_norm
                    prob /= np.nanmax(prob)
                    if plot_filled:
                        # xs = np.array(xs).reshape(prob.shape)
                        # ys = np.array(ys).reshape(prob.shape)
                        x = get_flat_data_points(pj, uniques[j])
                        y = get_flat_data_points(pi, uniques[i])
                        xx,yy = np.meshgrid(x,y)
                        # print(xs, ys)
                        ax.pcolormesh(xx, yy, prob, 
                                           cmap = cmap, norm = norm, shading = 'flat')
                        # ax.scatter(xs,ys,c=prob.flatten(), marker='o', cmap=cmap, norm=norm,
                        #                 edgecolor='k')
                        # im = ax.imshow(prob, cmap=cmap, norm=norm,
                        #                extent=(xs.min(), xs.max(), ys.max(), ys.min()),
                        #                interpolation='nearest', origin='lower', aspect='auto')
                        
                    else:
                        im = ax.scatter(xs,ys, c=prob.flatten(), cmap=cmap, norm=norm)
                
            else:
                ax.axis('off')
    # plt.minorticks_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pad = 0.01
    cbar = f.colorbar(sm, ax=axs[-1,:], pad=pad, #extend='min',
                      fraction=frac)
    # midpt = 10**(int(np.log10(vmin))//2)
    # print(midpt)
    # cbar.set_ticks([1e-3, 1e-2, 1e-1, 1])
    # cbar.set_ticklabels([r'$<10^{-3}$',r'$10^{-2}$' ,r'$10^{-1}$', r'$1$'])
    # cbar.set_ticks([1e-2, 1e-1, 1])
    # cbar.set_ticklabels([r'$<10^{-2}$' ,r'$10^{-1}$', r'$1$'])

    # cbar.ax.set_yticklabels([r'$<0.2$', r'$2$', r'$>20$']) 
    f.colorbar(sm, ax=axs[0,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[1,:], pad=pad, fraction=frac).ax.set_visible(False)
    f.colorbar(sm, ax=axs[2,:], pad=pad, fraction=frac).ax.set_visible(False)
    if z is None:
        # plt.savefig('plots/prob_plots/'+fn_root+'_triangle_like.pdf', bbox_inches='tight')
        plt.savefig(fn_root+'_triangle_like.pdf', bbox_inches='tight')
    else:
        plt.savefig('plots/prob_plots/'+fn_root+'_triangle_like_z{}.pdf'.format(z), 
            bbox_inches='tight')
    plt.close('all')


# def test_seaborn_kde(data):
#     like = data['like'] / data['like'].max()
#     for bw in [0.3,0.35,0.4,0.45]:
#         print(bw)
#             # p = sns.displot(
#             #     data=data, x="velocityOutflow", y="alphaOutflow", weights='like', kind='kde', fill=True,
#             #     levels = [0.68, 0.95, 0.99], cut=0, bw_adjust=1.0
#             # )
#         # p = sns.jointplot(
#         #     data=data, x="velocityOutflow", y="alphaOutflow", kind='kde', fill=True,
#         #     levels = [0.68, 0.95, 0.99], cmap='crest', 
#         #     bw_adjust=bw, weights=data['like'], cut=0,
#         #     # joint_kws={'bw_adjust':bw, 'cut':0, 'weights':data['like']}
#         #     marginal_kws={'weights':data['like'],'cut':0}
#         # )
#         p = sns.jointplot(
#             data=data, x="velocityOutflow", y="alphaOutflow", kind='hist', fill=True,
#             cmap='Blues', bins=20, weights=like, 
#             marginal_kws={'weights':like,'bins':20, 'kde':True, 'kde_kws':{'bw_adjust':bw}}
#             # kde_kws={'bw_adjust':bw, 'cut':0, 'weights':data['like'],
#             #            'levels':[0.68, 0.95, 0.99]}
#         )
#         p.plot_joint(sns.kdeplot, weights=like, cmap='Blues',
#                      bw_adjust=bw, cut=0, levels=[0.68, 0.95, 0.99])
#         # p.plot_marginals(sns.kdeplot, weights=data['like'], color='Blue',
#         #              bw_adjust=bw, cut=0)
        
#         p.set_axis_labels(r'$V_{\mathrm{outflow}}$', r'$\alpha_{\mathrm{outflow}}$')

#         # p = sns.jointplo
#         plt.savefig(f'test_sns_kde_bw{bw}.pdf')
#         plt.close('all')


# def test_skl_kde(data):
#     like = data['like'] / data['like'].max()
#     # for bw in [0.5,0.75,1.0,1.25,1.5]:
#     X = np.array([data['velocityOutflow'].to_numpy(),data['alphaOutflow'].to_numpy()]).T
#     print(X.shape)
#     for kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential']:
#         for bw in [0.5,1.0,1.5]:
#             kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(X)
#             plt.savefig(f'test_skl_kde_kernel{kernel}_bw{bw}.pdf')
#             plt.close('all')


def np_to_df(data, parameters, pvalues):
    data = data[~np.isnan(data).any(axis=1)]
    indices = data[:,0].astype(int)
    df = pd.DataFrame(data[:,1], columns=['loglike'], index=indices)
    param_array = np.zeros((len(data),len(parameters)))
    x = 0
    for i, values in enumerate(itertools.product(*pvalues)):
        for j, p in enumerate(parameters):
            if i in df.index:
                param_array[i-x, j] = values[j]
            else:
                print(f'missing data for i={i}')
                x+=1
                break
    for j, p in enumerate(parameters):
        df[p] = param_array[:,j]
    # print(df.head)
    return df

# filenames = ['ngdeep_vout_r2', 'ngdeep_tau_r2', 'fixed_ngdeep_params']
dfs = []

fn_root = 'final_params5'
# for fn_root in filenames:
yaml_fn = 'yamls/' + fn_root + '.yaml'
pkl_fn = fn_root + '.pkl'
data_fn = fn_root+'_results.npy'

with open(yaml_fn, 'r') as f:
    astro_params = yaml.safe_load(f)

pvalues = []
parameters = []
for k,p in astro_params['parameters'].items():
    if not p['sample']:
        values = [p['value']]
    elif p['sample']=='lin':
        values = np.linspace(p['min'], p['max'], p['nv'])
    elif p['sample']=='log':
        values = np.geomspace(p['min'], p['max'], p['nv'])
    else:
        raise Exception('Unknown value for sample')

    # pvalues.append(np.log10(values))
    pvalues.append(values)
    parameters.append(k)

nparam = len(parameters)
fn_root = 'final'

df_fns = [
    'final_params5.csv', 
    'final_params6.csv',
    'final_params7.csv',
    'final_params8.csv',
    'nr1.csv',
    'nr2.csv',
    'nr3.csv',
    'nr4.csv',
    'nr5.csv',
    'nr6.csv',
    'nr7.csv',
    'nr8.csv',
    'nr9.csv',
    'nr10.csv',
    'nr11.csv',
    'nr12.csv'
    ]

# df_fns = [
#     'cv_final_params5.csv', 
#     'cv_final_params6.csv',
#     'cv_final_params7.csv',
#     'cv_final_params8.csv',
#     'cv_nr1.csv',
#     'cv_nr2.csv',
#     'cv_nr3.csv',
#     'cv_nr4.csv'
#     ]

dfs = []
for fn in df_fns:
    dfi = pd.read_csv(fn)
    idx = fn.rfind('.')
    tag = fn[:idx]
    dfi.insert(0, 'tag', [tag]*len(dfi))
    dfs.append(dfi)
df = pd.concat(dfs, ignore_index=True)
df = df[
    (df['velocityOutflow']>=50) & (df['velocityOutflow']<=500) &
    (df['alphaOutflow']>=0.124) & (df['alphaOutflow']<=2.5) &
    (df['timescale']>=0.1) & (df['timescale']<=1.0) &
    (df['alphaStar']>-4.5) & (df['alphaStar']<=-0.5)
]
# df = df[(df['alphaOutflow']>=0.5) & (df['alphaOutflow']<=2.5)]
# df = df[(df['timescale']>=0.1) & (df['timescale']<=1.0)]
# df = df[(df['alphaStar']>-4.5) & (df['alphaStar']<=-0.5)]




df['loglike'].fillna(df['loglike'].min(skipna=True), inplace = True)
df.insert(len(df.columns), 'like', np.exp(df['loglike']))
df = df.sort_values('loglike', ascending=False)
df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
print(df.head(n=10))
# print(df['Unnamed: 0'])

# df.to_csv('final_df.csv')

fn_root = 'final'
# plot_astro_like('binned_'+fn_root, df, parameters,  None, True)
test_seaborn_kde(df)
# plot_astro_like(fn_root, df, parameters,  None, True)


# uniques = [np.sort(np.unique(df[p])) for p in parameters]
# like = np.exp(df['loglike'].values)
# tab_prob_1d = tabulate_1d(df, uniques, parameters, like)
# i = parameters.index('velocityOutflow')
# # j = parameters.index('alphaOutflow')
# vouts = uniques[i]


    


# data = np.array([vouts,alphaouts,like2d]).T
# new_df = pd.DataFrame(data, columns=['Voutflow', 'alphaOutflow', 'like2d'])
# new_df.to_csv('2d_df.csv')


# getting just alpha_out, V_out
# uniques = [np.sort(np.unique(df[p])) for p in parameters]
# like = np.exp(df['loglike'].values)
# tab_prob_2d, prob_norm = tabulate_2d(df, uniques, parameters, like)
# i = parameters.index('velocityOutflow')
# j = parameters.index('alphaOutflow')
# vouts = []
# alphaouts = []
# like2d = []
# for vi in uniques[i]:
#     for vj in uniques[j]:
#         vouts.append(vi)
#         alphaouts.append(vj)
#         like2d.append(tab_prob_2d[i][j])
# data = np.array([vouts,alphaouts,like2d]).T
# new_df = pd.DataFrame(data, columns=['Voutflow', 'alphaOutflow', 'like2d'])
# new_df.to_csv('2d_df.csv')