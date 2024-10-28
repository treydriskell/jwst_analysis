
import h5py
import numpy as np
import os
import os.path as path
# import pickle as pkl
# import matplotlib as mpl
# mpl.use('pdf')
# import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM
import itertools
import yaml 
# import corner
import pandas as pd
# from scipy.cluster.vq import kmeans
import glob
from multiprocessing import Pool
# import pickle 
import xml.etree.ElementTree as ET

global cosmo
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


def load_data(data_dir):
    data_fn = path.join(data_dir, 'data.npy')
    if path.isfile(data_fn):
        data = np.load(data_fn)
    else:
        app_mags = []
        abs_mags = []
        logmhs = []
        zs = []
        for z in ['8.0', '12.0', '16.0']: #
        # for z in ['8.0']:
            print(z)
            # outfiles = glob.glob(data_dir+f'z{z}:MPI*.hdf5')
            outfiles = glob.glob(data_dir+f'z{z}.hdf5')
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


Vout_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
alphaOut_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
tau0_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/timescale'
alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/exponentVelocity'

def run(data_dir):
    # print(f'Starting p{i}')
    # base = '/data001/gdriskell/jwst_blitz_astro_samples/'
    # data_dir = path.join(base, dirname+f'_p{i}/')
    # data_dir = path.join(base, dirname)
    xml_fn = path.join(data_dir,'z8.0.xml')
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    Vout = float(root.find(Vout_xml).get('value'))
    alphaOut = float(root.find(alphaOut_xml).get('value'))
    if 'nr' in data_dir:
        tau0_xml = 'starFormationRateDisks/starFormationTimescale/timescale'
        alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/exponentVelocity'
    else:
        tau0_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/timescale'
        alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/starFormationTimescale/exponentVelocity'

    timescale = float(root.find(tau0_xml).get('value'))
    alphaStar = float(root.find(alphaStar_xml).get('value'))

    # if (Vout >= 100) and (Vout <=200):
    #     if (alphaOut >= 0.5) and (alphaOut <=2.5):
    #         if (timescale >= 0.1) and (timescale <=1.0):
    #             if (alphaStar > -4.5) and (alphaStar <=-0.5):
    # try:
    try:
        weights, logmhs, abs_mags, app_mags, zs = load_data(data_dir).T
        # Vout = [Vout] * len(logmhs) 
        # alphaOut = [alphaOut] * len(logmhs)
        # timescale = [timescale] * len(logmhs)
        # alphaStar = [alphaStar] * len(logmhs)
        # data = np.stack([Vout, alphaOut, timescale, alphaStar, [logmhs], [zs], [app_mags], [abs_mags]],axis=0).T
        data = [[Vout, alphaOut, timescale, alphaStar, logmhs, zs, app_mags, abs_mags]]
        df = pd.DataFrame(data, columns=['Vout', 'alphaOut', 'tau0', 'alphaStar', 'logMh', 'z', 'muv','Muv'])
    except: 
        return None
    return df #[Vout, alphaOut, timescale, alphaStar, logmhs, zs, app_mags, abs_mags]
    # except:
    #     return None
    # return None

    
    # idx = logmhs > 9
    # logmhs = logmhs[idx]
    # zs = zs[idx]
    # app_mags = app_mags[idx]
    # abs_mags = abs_mags[idx]

    # Vout = [root.find(Vout_xml).get('value')]*len(logmhs)
    # alphaOut = [root.find(alphaOut_xml).get('value')]*len(logmhs)
    # tau0 = [root.find(tau0_xml).get('value')]*len(logmhs)
    # alphaStar = [root.find(alphaStar_xml).get('value')]*len(logmhs)
   
    # print(i,Vout, alphaOut, tau0, alphaStar)
    # data = np.stack([Vout, alphaOut, timescale, alphaStar, logmhs, zs, app_mags, abs_mags],axis=0).T
    # print(data.shape)
    # df = pd.DataFrame(data, columns=['Vout', 'alphaOut', 'tau0', 'alphaStar', 'logMh', 'z', 'muv','Muv'])
    # return df
    # return data


if __name__ == "__main__":

    base = '/data001/gdriskell/jwst_blitz_astro_samples/'
    dirnames = [
        'final_params5', 
        'final_params6',
        'final_params7',
        'final_params8',
        'nr1',
        'nr2',
        'nr3',
        'nr4',
        'nr5',
        'nr6',
        'nr7',
        'nr8',
        'nr9',
        'nr10',
        'nr11',
        'nr12'
    ]
    # dirnames = ['nr1']
    data = []
    for dir_fn in dirnames:
        # dir_fn = '/data001/gdriskell/jwst_blitz_astro'
        dir_fn = path.join(base, dir_fn)
        print(f'starting {dir_fn}')
        outfiles = glob.glob(dir_fn+f'_p*/')
        print(len(outfiles))
        for fn in outfiles:
            d = run(fn)
            if not (d is None):
                data.append(d)
            else:
                print(fn)
        print(f'finished {dir_fn}')
        
        
    df = pd.concat(data, ignore_index=True)
    # print(df.head(5))
    # print(df['z'])
    # print(len(df))
    df.to_csv('/data001/gdriskell/full_data_df.csv')
