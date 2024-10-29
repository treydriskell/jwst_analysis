# create_jwst_xmls.py

# import h5py
import numpy as np
import os
import os.path as path
import subprocess
import itertools
import xml.etree.ElementTree as ET
import yaml
# import pandas as pd


def create_xml(astro_params, outdir, values, param_paths, z):
    template_xml = 'xmls/timescale_template.xml' #astro_params['xml_template']
    tree = ET.parse(template_xml)
    root = tree.getroot()
    for j in range(len(values)):
        v = '{:.2e}'.format(values[j])
        param_path = param_paths[j]
        xml_ps = root.findall(param_path) 
        if len(xml_ps)==0:
            idx = param_path.rfind('/')
            parent = root.find(param_path[:idx])
            xml_p = ET.SubElement(parent, param_path[idx+1:], value=v)
        else:
            xml_ps[0].set('value', v) 
            # [p.set('value', v) for p in xml_ps]
    basez = root.find('mergerTreeConstructor/redshiftBase')
    # basez.set('value', '{:.1f}'.format(z))
    basez.set('value', z)
    outputFileName = root.find('outputFileName')
    # out_fn = path.join(outdir,"z_{:.1f}.hdf5".format(z))
    out_fn = path.join(outdir, 'z'+z+'.hdf5')
    outputFileName.set('value', out_fn)
    outTimes = root.find('outputTimes/redshifts')
    # outTimes.set('value', '{:.1f}'.format(z))
    outTimes.set('value', z)
    # xml_fn = path.join(outdir,'z_{:.1f}.xml'.format(z))
    xml_fn = path.join(outdir, 'z'+z+'.xml')
    tree.write(xml_fn)
    return xml_fn

# def create_job(astro_params, fn_base, xml_fn, z):
#     template_job = '/home1/gdriskel/jwst_analysis/jwst_template.job' #astro_params['job_template']
#     with open(template_job, 'rb') as f:
#         lines = f.readlines()
#     lines[6] = '#SBATCH --job-name=' + fn_base + '\n'
#     lines[6] = '#SBATCH --output=/scratch1/gdriskel/jwst_data/jobs/' + fn_base + '.out\n'
#     lines[15] = '/home1/gdriskel/galacticus/galacticus.exe ' + xml_fn + '\n'

#     # job_fn = path.join(astro_params['job_dir'], fn_base + '.pbs')
#     job_fn = path.join('/scratch1/gdriskel/jwst_data/jobs/', fn_base + '.job')
#     with open(job_fn, 'wb') as f:
#         f.writelines(lines)
#     return job_fn


def create_jobs_from_list(astro_params, xml_fns, initial,final):
    template_job = '/home1/gdriskel/jwst_analysis/jwst_template.job' #astro_params['job_template']
    fn_base = astro_params['param_dir']+'_pi{}_pf{}'.format(initial,final)
    with open(template_job, 'r') as f:
        lines = f.readlines()
    lines[6] = '#SBATCH --job-name=' + fn_base + '\n'
    lines[7] = '#SBATCH --output=/project/gluscevi_339/jwst_data/jobs/' + fn_base + '.out\n'
    lines[8] = '#SBATCH --error=/project/gluscevi_339/jwst_data/jobs/' + fn_base + '.err\n'
    for xml_fn in xml_fns:
        #lines.append('mpirun --n 16 --map-by node --bind-to none -mca btl ^openib /home/gdriskell/galacticus/Galacticus.exe ' + xml_fn + '\n')
        # mpirun --n ??? --map-by node --bind-to none -mca btl ^openib
        right = xml_fn.rfind('.') 
        out_fn = xml_fn[:right]+'.out'
        # lines.append('mpirun --n 160 --map-by node --bind-to none -mca btl ^openib /home/gdriskell/galacticus/Galacticus.exe '+ xml_fn + ' &> ' + out_fn  +'\n')
        lines.append('/home1/gdriskel/galacticus/galacticus.exe '+ xml_fn + ' &> ' + out_fn  +'\n')
    lines.append('echo "job ended at `date`" \n')
    lines.append('exit\n')
    # job_fn = path.join(astro_params['job_dir'], fn_base + '.sbatch')
    job_fn = path.join('/project/gluscevi_339/jwst_data/jobs/', fn_base + '.job')
    # print(job_fn)
    with open(job_fn, 'w') as f:
        # print(type(f),type(lines)) 
        f.writelines(lines)
    return job_fn

# yaml_fns = [f'yamls/nr{i}.yaml' for i in range(12)]
# yaml_fns += [f'yamls/final_params{i}.yaml' i in range(5,10)]
yaml_fns = ['yamls/test_params.yaml']

for yaml_fn in yaml_fns:#'yamls/nr12_params.yaml' # 'yamls/maxlike_corners.yaml' #'simplified.yaml', 'updated_timescale_params.yaml' #'z0_params.yaml' #'baugh2005_timescale_params2.yaml'#'timescale_sfr_params.yaml' # 'vary_astro_params.yaml'
    with open(yaml_fn, 'r') as f:
        astro_params = yaml.safe_load(f)

    pvalues = []
    parameters = []
    param_paths = []
    for k,p in astro_params['parameters'].items():
        if not p['sample']:
            values = [p['value']]
        elif p['sample']=='lin':
            values = np.linspace(p['min'], p['max'], p['nv'])
        elif p['sample']=='log':
            values = np.geomspace(p['min'], p['max'], p['nv'])
        else:
            raise Exception('Unknown value for sample')
        pvalues.append(values)
        parameters.append(k)
        param_paths.append(p['path'])


    params_per_job = 1

    # rerun_indices = [376, 540, 618, 990]
    metadata = []

    xml_fns = []
    jobids = []
    for i, values in enumerate(itertools.product(*pvalues)):
        # if i==931:
        j=i
        # if i<1:
        # j = 1225+i
        # ji=j
        if i==0:
            ji=j

        # print(j)
        fn_base = astro_params['param_dir']+'_p{}'.format(j)
        # outdir = path.join(astro_params['base_dir'], fn_base)
        outdir = path.join('/project/gluscevi_339/jwst_data/', fn_base)

        if not path.isdir(outdir):
            os.makedirs(outdir)
    # if i in rerun_indices:
        # j=i
        # ji=i
        # if j in reruns:
        # if True:
        for z in astro_params['zs']:
            # if not (z == '16.0'):
            xml_fn =  create_xml(astro_params, outdir, values, param_paths, z) #path.join(outdir, 'z'+z+'.xml') 
            xml_fns.append(xml_fn)
        if (j%params_per_job) == (params_per_job-1):
            # os.chdir(astro_params['job_dir'])
            os.chdir('/project/gluscevi_339/jwst_data/jobs/')
            job_fn = create_jobs_from_list(astro_params, xml_fns,ji,j)
            # if len(jobids)<4:
            print(job_fn)

            output = subprocess.check_output(['sbatch', job_fn], text=True)
                # jobid = output[:output.find('.')]
                # jobids.append(jobid)


            # else:
            #     dep_string ='depend=afterok:' + jobids.pop(0)
            #     output = subprocess.check_output(['qsub', '-W', dep_string, job_fn], text=True)
            #     jobid = output[:output.find('.')]
            #     print(jobid, dep_string)
            #     jobids.append(jobid)
            xml_fns = []
            ji=j+1
                
        # print(j)
    if len(xml_fns)>1:
        # print('bad')
        # os.chdir(astro_params['job_dir'])
        os.chdir('/project/gluscevi_339/jwst_data/jobs/')
        job_fn = create_jobs_from_list(astro_params, xml_fns,i-(i%params_per_job),i)
        os.system('sbatch ' + job_fn)
        # dep_string ='depend=afterok:' + jobids.pop(0)
        # output = subprocess.check_output(['qsub', '-W', dep_string, job_fn], text=True)
        # jobid = output[:output.find('.')]
        # jobids.append(jobid)

